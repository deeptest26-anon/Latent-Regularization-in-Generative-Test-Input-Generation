import sys, copy, traceback, numpy as np
import torch, torch.fft, torch.nn
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import stylegan.legacy
import config


# ------------------------------------------------------------
class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            msg = str(value) if isinstance(value, CapturedException) else traceback.format_exc()
        super().__init__(msg)

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

# ------------------------------------------------------------
def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4,
                                       cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat  = torch.as_tensor(mat).to(torch.float32)
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f  = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w  = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real
    f  = f * w

    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0, 1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0, 2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)
    f   = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    p   = f.shape[0] // 2

    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g     = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear',
                                        padding_mode='zeros', align_corners=False)

    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest',
                                        padding_mode='zeros', align_corners=False)
    return z, m

# ------------------------------------------------------------
class Renderer:
    def __init__(self, disable_timing=False):
        self._device   = torch.device(config.DEVICE)
        self._dtype    = torch.float32
        self._pkl_data = {}
        self._networks = {}
        self._cmaps    = {}
        self._is_timing = False
        self._disable_timing = disable_timing
        self._net_layers = {}

    # --------------------------------------------------------
    def render(self, **args):
        self._is_timing = False
        res = dnnlib.EasyDict()
        try:
            self._render_impl(res, **args)
        except Exception:
            res.error = CapturedException()
        if 'stats' in res:
            res.stats = res.stats.cpu().numpy()
        if 'error' in res:
            res.error = str(res.error)
        return res

    # --------------------------------------------------------
    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = stylegan.legacy.load_network_pkl(f)
                print('Done.')
            except Exception:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net  = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net       = self._networks.get(cache_key)
        if net is None:
            try:
                net = copy.deepcopy(orig_net)
                net = self._tweak_network(net, **tweak_kwargs)
                net.to(self._device)
            except Exception:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    # --------------------------------------------------------
    def _tweak_network(self, net):
        return net  

    def to_device(self, buf): return buf.to(self._device)
    def to_cpu   (self, buf): return buf.cpu()
    def _ignore_timing(self): self._is_timing = False

    # --------------------------------------------------------
    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, 1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x  = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        return torch.nn.functional.embedding(x, cmap)

    # --------------------------------------------------------
    def _render_impl(self, res,
        pkl=None,
        w0_seeds=[[0, 1]],
        w_load=None,
        w_load_seed=None,
        class_idx=None,
        mixclass_idx=None,
        stylemix_idx=[],
        stylemix_seed=None,
        trunc_psi=None,
        trunc_cutoff=None,
        random_seed=0,
        noise_mode='random',
        force_fp32=False,
        layer_name=None,
        sel_channels=3,
        base_channel=0,
        img_scale_db=0,
        img_normalize=False,
        to_pil=False,
        input_transform=None,
        untransform=False,
        INTERPOLATION_ALPHA=1.0,
    ):
        # load generator 
        G = self.get_network(pkl, 'G_ema')
        self.G = G
        res.img_resolution = G.img_resolution
        res.num_ws         = G.num_ws
        res.has_noise      = any('noise_const' in n for n, _ in G.synthesis.named_buffers())
        res.has_input_transform = hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform')

        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # build conditioning vectors 
        stylemix_cs = None
        w0_seeds_only = [seed for seed, _ in w0_seeds]
        if stylemix_seed is not None:
            all_seeds = w0_seeds_only + [stylemix_seed]
            stylemix_cs = np.zeros([1, G.c_dim], np.float32)
            if G.c_dim > 0:
                if mixclass_idx is not None:
                    stylemix_cs[:, mixclass_idx] = 1
                else:
                    rnd = np.random.RandomState(stylemix_seed)
                    stylemix_cs[:, rnd.randint(G.c_dim)] = 1
        else:
            all_seeds = w0_seeds_only

        w0_cs = np.zeros([len(w0_seeds), G.c_dim], np.float32)
        if G.c_dim > 0:
            if class_idx is not None:
                if isinstance(class_idx, list):
                    for i, _ in enumerate(w0_seeds):
                        w0_cs[i, class_idx[i]] = 1
                else:
                    w0_cs[:, class_idx] = 1
            else:
                for i, (seed, _) in enumerate(w0_seeds):
                    rnd = np.random.RandomState(seed)
                    w0_cs[i, rnd.randint(G.c_dim)] = 1

        all_cs = np.concatenate([w0_cs, stylemix_cs], axis=0) if stylemix_cs is not None else w0_cs

        # sample z 
        all_zs = np.zeros([len(all_seeds), G.z_dim], np.float32)
        for i, seed in enumerate(all_seeds):
            rnd = np.random.RandomState(seed)
            all_zs[i] = rnd.randn(G.z_dim)

        # map to w 
        w_avg  = G.mapping.w_avg
        all_zs = self.to_device(torch.from_numpy(all_zs))
        all_cs = self.to_device(torch.from_numpy(all_cs))
        if w_load is not None:
            w_load = self.to_device(torch.from_numpy(w_load)).squeeze(0)

        all_ws = G.mapping(z=all_zs, c=all_cs,
                           truncation_psi=trunc_psi,
                           truncation_cutoff=trunc_cutoff) - w_avg
        all_ws = dict(zip(all_seeds, all_ws))

        # w‑override
        if w_load is not None:
            for seed in w0_seeds_only:
                if w_load_seed is None or seed == w_load_seed:
                    all_ws[seed] = w_load

        # linear blend of w vectors
        w = torch.stack([all_ws[seed] * weight for seed, weight in w0_seeds]).sum(0, keepdim=True)

        # style‑mix
        stylemix_idx = [i for i in stylemix_idx if 0 <= i < G.num_ws]
        if stylemix_seed is not None and stylemix_idx:
            w2 = all_ws[stylemix_seed][None, :]
            for i in stylemix_idx:
                w[:, i] = (1 - INTERPOLATION_ALPHA) * w[:, i] + INTERPOLATION_ALPHA * w2[:, i]

        if w_load is None:
            w += w_avg

        # synthesize
        synth_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32)
        torch.manual_seed(random_seed)
        out, layers = self.run_synthesis_net(G.synthesis, w, capture_layer=layer_name, **synth_kwargs)

        cache_key = (G.synthesis, tuple(sorted(synth_kwargs.items())))
        if cache_key not in self._net_layers:
            if layer_name is not None:
                torch.manual_seed(random_seed)
                _out, layers = self.run_synthesis_net(G.synthesis, w, **synth_kwargs)
            self._net_layers[cache_key] = layers

        # undo StyleGAN‑input transform 
        if untransform and res.has_input_transform:
            out, _ = _apply_affine_transformation(out.to(torch.float32),
                                                  G.synthesis.input.transform, amax=6)

        # choose channels & stats
        out = out[0].to(torch.float32)
        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel_val = max(min(base_channel, out.shape[0] - sel_channels), 0)
        sel = out[base_channel_val: base_channel_val + sel_channels]
        sel = sel.clamp(-1.0, 1.0)         

        res.stats = torch.stack([
            out.mean(), sel.mean(),
            out.std(),  sel.std(),
            out.norm(float('inf')), sel.norm(float('inf')),
        ])

        # convert to uint8 image
        img = sel
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)

        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            if img.shape[2] == 1:
                img = img.squeeze()
            img = Image.fromarray(img)
        res.image = img
        res.w     = w.detach().cpu().numpy()

    # --------------------------------------------------------
    @staticmethod
    def run_synthesis_net(net, *args, capture_layer=None, **kwargs):
        submodule_names = {m: n for n, m in net.named_modules()}
        unique_names = set()
        layers = []

        def _hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [o for o in outputs if isinstance(o, torch.Tensor) and o.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5:
                    out = out.mean(2)
                name = submodule_names[module] or 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                layers.append(dnnlib.EasyDict(name=name,
                                              shape=[int(x) for x in out.shape],
                                              dtype=str(out.dtype).split('.')[-1]))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [m.register_forward_hook(_hook) for m in net.modules()]
        try:
            out = net(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for h in hooks:
            h.remove()
        return out, layers
