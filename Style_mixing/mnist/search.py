# mnist/search.py
import os
import copy
import json
import time
import csv
import numpy as np
import sys
from PIL import Image
import traceback


from PIL import ImageDraw, ImageFont


import torch
try:
    import lpips
    _LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    _LPIPS_AVAILABLE = False
    print("[warn] lpips not installed; JSON will include lpips: null. Try: pip install lpips")

_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_LPIPS_MIN_SIDE = 64  

def _get_lpips_model():
  
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if _LPIPS_MODEL is None and _LPIPS_AVAILABLE:
        dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        _LPIPS_DEVICE = torch.device(dev)
        _LPIPS_MODEL = lpips.LPIPS(net='alex').to(_LPIPS_DEVICE).eval()
    return _LPIPS_MODEL, _LPIPS_DEVICE

def _lpips_base_vs(arr_base_uint8_hw, arr_other_uint8_hw):
    
    if not _LPIPS_AVAILABLE:
        return None
    model, device = _get_lpips_model()
    if model is None:
        return None

    x = torch.from_numpy(arr_base_uint8_hw).float() / 255.0
    y = torch.from_numpy(arr_other_uint8_hw).float() / 255.0
    if x.ndim == 2:
        x = x.unsqueeze(0)  
    if y.ndim == 2:
        y = y.unsqueeze(0)


    x = x.repeat(3, 1, 1).unsqueeze(0)  
    y = y.repeat(3, 1, 1).unsqueeze(0)  

    H, W = x.shape[-2], x.shape[-1]
    tgt_h = max(H, _LPIPS_MIN_SIDE)
    tgt_w = max(W, _LPIPS_MIN_SIDE)
    if (H != tgt_h) or (W != tgt_w):
        x = torch.nn.functional.interpolate(x, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
        y = torch.nn.functional.interpolate(y, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)


    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0

    with torch.no_grad():
        d = model(x.to(device), y.to(device))
    return float(d.item())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation


ENV_TOPK_TARGETS = int(os.getenv("TOPK_TARGETS", 2))                
ENV_SINGLE_LAYER_ONLY = bool(int(os.getenv("SINGLE_LAYER_ONLY", 1)))  
ENV_MAX_SECONDS_PER_SEED = int(os.getenv("MAX_SECONDS_PER_SEED", 30))
ENV_MAX_RENDERS_PER_SEED = int(os.getenv("MAX_RENDERS_PER_SEED", 400))


DATASET_NAME = "mnist"
METRICS_CSV = os.path.join(FRONTIER_PAIRS, f"{DATASET_NAME}_metrics.csv")
CSV_FIELDS = [
    "dataset","class_idx","w0_seed","psi","phase","accepted",
    "pred_top1","pred_top2","conf_top1","margin","entropy",
    "stylemix_seed","stylemix_layer","alpha","alpha_iters",
    "ssi","l2_distance","img_l2","m_img_l2","runtime_ms","timestamp"
]

def _ensure_csv():
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

def _append_row(row):
    try:
        _ensure_csv()
        for k in CSV_FIELDS:
            row.setdefault(k, "")
        with open(METRICS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
    except Exception as e:
        print(f"[warn] CSV log failed: {e}")

def _crop28(pil_img):
    img = pil_img.crop((2, 2, pil_img.width - 2, pil_img.height - 2))
    return np.array(img)

def _predict(arr_uint8_hw, label):
    accepted, confidence, predictions = Predictor().predict_datapoint(
        np.reshape(arr_uint8_hw, (-1, 28, 28, 1)), label
    )
    preds = np.asarray(predictions, dtype=np.float64)
    s = preds.sum() + 1e-12
    p = preds / s
    top1 = int(np.argmax(p))
    top2 = int(np.argsort(-p)[1])
    margin = float(p[top1] - p[top2])
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return bool(accepted), float(confidence), p, top1, top2, margin, entropy

def _log_error(context, exc):
    try:
        os.makedirs(FRONTIER_PAIRS, exist_ok=True)
        with open(os.path.join(FRONTIER_PAIRS, "errors.log"), "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {context}\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    except Exception:
        pass


SAFE_LAYERS = []
for l in STYLEMIX_LAYERS:
    if isinstance(l, (list, tuple)):
        if len(l) == 1:
            SAFE_LAYERS.append([int(l[0])])
        else:
            SAFE_LAYERS.append([int(x) for x in l])
    elif isinstance(l, int):
        SAFE_LAYERS.append([l])

class mimicry:
    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT, step_size=1,
                 forced_trunc_psi=None, no_adaptive=False,
                 topk_targets=ENV_TOPK_TARGETS, single_layer_only=ENV_SINGLE_LAYER_ONLY,
                 max_seconds_per_seed=ENV_MAX_SECONDS_PER_SEED, max_renders_per_seed=ENV_MAX_RENDERS_PER_SEED):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = SAFE_LAYERS
        self.step_size = step_size

   
        self.forced_trunc_psi = forced_trunc_psi
        self.no_adaptive = no_adaptive


        self.topk_targets = int(topk_targets)
        self.single_layer_only = bool(single_layer_only)
        self.max_seconds_per_seed = int(max_seconds_per_seed)
        self.max_renders_per_seed = int(max_renders_per_seed)

        self.state['renderer'] = Renderer()

    def _iter_layers(self):
        if self.single_layer_only:
            for layer in self.layers:
                if len(layer) == 1:
                    yield layer
        else:
            for layer in self.layers:
                yield layer

    def render_state(self, state=None):
        if state is None:
            state = self.state
        t0 = time.time()
        try:
            result = state['renderer'].render(
                pkl=INIT_PKL,
                w0_seeds=state['params']['w0_seeds'],
                class_idx=state['params']['class_idx'],
                mixclass_idx=state['params']['mixclass_idx'],
                stylemix_idx=state['params']['stylemix_idx'],
                stylemix_seed=state['params']['stylemix_seed'],
                trunc_psi=state['params']['trunc_psi'],
                trunc_cutoff=state['params']['trunc_cutoff'],
                img_normalize=state['params']['img_normalize'],
                to_pil=state['params']['to_pil'],
                INTERPOLATION_ALPHA=state['params'].get('INTERPOLATION_ALPHA', 1.0),
            )
        except Exception as e:
            _log_error(f"render_state failed | params={state['params']}", e)
            return {"error": str(e)}, copy.deepcopy(state['params'])

        self._render_calls = getattr(self, "_render_calls", 0) + 1
        info = copy.deepcopy(state['params'])
        info["_runtime_ms"] = int((time.time() - t0) * 1000)
        return result, info

    def search(self):
        root = f"{FRONTIER_PAIRS}/{self.class_idx}/"
        frontier_seed_count = 0
        tolerance = 1e-10

        while frontier_seed_count < self.search_limit:
            seed_t0 = time.time()
            self._render_calls = 0

            def _budget_exceeded():
                return ((time.time() - seed_t0) > self.max_seconds_per_seed) or (self._render_calls >= self.max_renders_per_seed)

            state = self.state
            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None
            state["params"]["INTERPOLATION_ALPHA"] = 1.0

         
            state['params']['trunc_psi'] = 1.0
            state['params']['trunc_cutoff'] = None
            if self.forced_trunc_psi is not None:
                state['params']['trunc_psi'] = float(self.forced_trunc_psi)

            digit, digit_info = self.render_state()

            if 'image' not in digit:
                print(f"Render failed with error: {digit.get('error', 'Unknown error')}")
                self.w0_seed += self.step_size
                continue

            label = digit_info["class_idx"]
            image = digit['image']
            image_array = _crop28(image)

            accepted, confidence, predictions, top1, top2, margin, entropy = _predict(image_array, label)

            digit_info["accepted"] = bool(accepted)
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()
            digit_info["trunc_psi"] = float(state['params']['trunc_psi'])

            _append_row({
                "dataset": DATASET_NAME, "class_idx": self.class_idx, "w0_seed": self.w0_seed,
                "psi": digit_info["trunc_psi"], "phase": "baseline",
                "accepted": int(accepted), "pred_top1": top1, "pred_top2": top2,
                "conf_top1": float(confidence), "margin": float(margin), "entropy": float(entropy),
                "stylemix_seed": "", "stylemix_layer": "", "alpha": "", "alpha_iters": "",
                "ssi": "", "l2_distance": "", "img_l2": "", "m_img_l2": "",
                "runtime_ms": digit_info.get("_runtime_ms",""), "timestamp": int(time.time())
            })

            if not accepted:
                if self.forced_trunc_psi is not None and self.no_adaptive:
                    self.w0_seed += self.step_size
                    continue

                truncation_values = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
                accepted = False
                for trunc_psi in truncation_values:
                    if _budget_exceeded():
                        break
                    state['params']['trunc_psi'] = trunc_psi
                    digit, digit_info = self.render_state()
                    if 'image' not in digit:
                        print(f"Render failed with error: {digit.get('error', 'Unknown error')} at trunc_psi={trunc_psi}")
                        continue
                    image_array = _crop28(digit['image'])
                    accepted, confidence, predictions, top1, top2, margin, entropy = _predict(image_array, label)
                    digit_info["accepted"] = bool(accepted)
                    digit_info["exp-confidence"] = float(confidence)
                    digit_info["predictions"] = predictions.tolist()
                    digit_info["trunc_psi"] = float(trunc_psi)
                    _append_row({
                        "dataset": DATASET_NAME, "class_idx": self.class_idx, "w0_seed": self.w0_seed,
                        "psi": float(trunc_psi), "phase": "baseline(adapt)",
                        "accepted": int(accepted), "pred_top1": top1, "pred_top2": top2,
                        "conf_top1": float(confidence), "margin": float(margin), "entropy": float(entropy),
                        "stylemix_seed": "", "stylemix_layer": "", "alpha": "", "alpha_iters": "",
                        "ssi": "", "l2_distance": "", "img_l2": "", "m_img_l2": "",
                        "runtime_ms": digit_info.get("_runtime_ms",""), "timestamp": int(time.time())
                    })
                    if accepted:
                        break

                if not accepted:
                    self.w0_seed += self.step_size
                    continue

                state['params']['trunc_psi'] = digit_info["trunc_psi"]

            order = np.argsort(-predictions)
            targets = [int(i) for i in order if i != label and predictions[i] > 0][:max(0, self.topk_targets)]

            if len(targets) > 0:
                found_at_least_one = False
                for stylemix_cls in targets:
                    found_mutation = False
                    self.stylemix_seed = 0

                    while (not found_mutation) and (self.stylemix_seed < self.stylemix_seed_limit) and (not _budget_exceeded()):

                        if self.stylemix_seed == self.w0_seed:
                            self.stylemix_seed += 1
                        state["params"]["stylemix_seed"] = self.stylemix_seed

                        for idx, layer in enumerate(self._iter_layers()):
                            if _budget_exceeded():
                                break
                            state["params"]["stylemix_idx"] = layer

                            state["params"]["INTERPOLATION_ALPHA"] = 1.0
                            m_digit, m_digit_info = self.render_state()
                            if 'image' not in m_digit:
                                print(f"Render failed with error: {m_digit.get('error', 'Unknown error')}")
                                continue

                            m_image_array = _crop28(m_digit['image'])
                            m_accepted, m_conf, m_predictions, m_top1, _, _, _ = _predict(m_image_array, label)
                            m_class = m_top1

                            if m_class != label:
                                alpha_min, alpha_max = 0.0, 1.0
                                iteration, max_iterations = 0, 20
                                last_correct_image = None
                                last_correct_alpha = None
                                last_correct_predictions = None
                                last_correct_confidence = None

                                while iteration < max_iterations and (alpha_max - alpha_min) > 1e-10 and (not _budget_exceeded()):
                                    alpha = (alpha_min + alpha_max) / 2.0
                                    state["params"]["INTERPOLATION_ALPHA"] = alpha
                                    m_digit, m_digit_info = self.render_state()
                                    if 'image' not in m_digit:
                                        print(f"Render failed with error: {m_digit.get('error', 'Unknown error')}")
                                        break
                                    m_image_array = _crop28(m_digit['image'])
                                    m_accepted, m_conf, m_predictions, m_top1, _, _, _ = _predict(m_image_array, label)
                                    m_class = m_top1
                                    if m_accepted:
                                        alpha_min = alpha
                                        last_correct_image = m_image_array.copy()
                                        last_correct_alpha = alpha
                                        last_correct_predictions = m_predictions.copy()
                                        last_correct_confidence = m_conf
                                    else:
                                        alpha_max = alpha
                                    iteration += 1

                                if alpha_max != 1.0:
                                    state["params"]["INTERPOLATION_ALPHA"] = alpha_max
                                    m_digit, m_digit_info = self.render_state()
                                    if 'image' not in m_digit:
                                        print(f"Render failed with error: {m_digit.get('error', 'Unknown error')}")
                                        continue
                                    m_image_array = _crop28(m_digit['image'])
                                    m_accepted, m_conf, m_predictions, m_top1, _, m_margin, m_entropy = _predict(m_image_array, label)
                                    m_class = m_top1

                                    if m_class == stylemix_cls:
                                        try:
                                            valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                                        except Exception as e:
                                            _log_error("validate_mutation failed", e)
                                            valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = False, 0.0, 0.0, 0.0, 0.0

                                        if valid_mutation:
                                            if not found_at_least_one:
                                                frontier_seed_count += 1
                                                found_at_least_one = True

                                            path = f"{root}{self.w0_seed}/"
                                            seed_name = f"0-{targets[0]}"
                                            img_path = f"{path}/{seed_name}.png"
                                            if not os.path.exists(img_path):
                                                os.makedirs(path, exist_ok=True)
                                                Image.fromarray(image_array.astype(np.uint8)).save(img_path)
                                                digit_info["l2_norm"] = float(img_l2)
                                                digit_info["lpips"] = 0.0 
                                                with open(f"{path}/{seed_name}.json", 'w') as f:
                                                    json.dump(digit_info, f, sort_keys=True, indent=4)

                                            found_mutation = True

                                            lpips_mis = _lpips_base_vs(image_array, m_image_array)

                                            if last_correct_image is not None:
                                                correct_img_uint8 = np.clip(last_correct_image, 0, 255).astype(np.uint8)
                                                correct_pil_image = Image.fromarray(correct_img_uint8)
                                                correct_img_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{last_correct_alpha:.6f}-correct.png"
                                                correct_pil_image.save(f"{path}/{correct_img_name}")

                                                correct_info = m_digit_info.copy()
                                                correct_info["alpha"] = float(last_correct_alpha)
                                                correct_info["accepted"] = True
                                                correct_info["predictions"] = last_correct_predictions.tolist()
                                                correct_info["exp-confidence"] = float(last_correct_confidence)
                                            
                                                lpips_cor = _lpips_base_vs(image_array, correct_img_uint8)
                                                correct_info["lpips"] = lpips_cor
                                                with open(f"{path}/{correct_img_name}.json", 'w') as f:
                                                    json.dump(correct_info, f, sort_keys=True, indent=4)

                                                _append_row({
                                                    "dataset": DATASET_NAME, "class_idx": self.class_idx, "w0_seed": self.w0_seed,
                                                    "psi": digit_info["trunc_psi"], "phase": "correct", "accepted": 1,
                                                    "pred_top1": int(np.argmax(last_correct_predictions)),
                                                    "pred_top2": int(np.argsort(-last_correct_predictions)[1]),
                                                    "conf_top1": float(last_correct_confidence),
                                                    "margin": "", "entropy": "",
                                                    "stylemix_seed": self.stylemix_seed,
                                                    "stylemix_layer": str(layer),
                                                    "alpha": float(last_correct_alpha), "alpha_iters": iteration,
                                                    "ssi": float(ssi), "l2_distance": float(l2_distance),
                                                    "img_l2": float(img_l2), "m_img_l2": float(m_img_l2),
                                                    "runtime_ms": m_digit_info.get("_runtime_ms",""),
                                                    "timestamp": int(time.time())
                                                })

                                            m_path = f"{path}/{stylemix_cls}"
                                            os.makedirs(m_path, exist_ok=True)
                                            m_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{alpha_max:.6f}-misclassified.png"
                                            m_digit_info["accepted"] = False
                                            m_digit_info["predicted-class"] = int(m_class)
                                            m_digit_info["exp-confidence"] = float(m_conf)
                                            m_digit_info["predictions"] = m_predictions.tolist()
                                            m_digit_info["ssi"] = float(ssi)
                                            m_digit_info["l2_norm"] = float(m_img_l2)
                                            m_digit_info["l2_distance"] = float(l2_distance)
                                            m_digit_info["alpha"] = float(alpha_max)
                                      
                                            m_digit_info["lpips"] = lpips_mis
                                            with open(f"{m_path}/{m_name}.json", 'w') as f:
                                                json.dump(m_digit_info, f, sort_keys=True, indent=4)
                                            Image.fromarray(np.clip(m_image_array, 0, 255).astype(np.uint8)).save(f"{m_path}/{m_name}.png")

                                            _append_row({
                                                "dataset": DATASET_NAME, "class_idx": self.class_idx, "w0_seed": self.w0_seed,
                                                "psi": digit_info["trunc_psi"], "phase": "flip", "accepted": 0,
                                                "pred_top1": int(m_class),
                                                "pred_top2": int(np.argsort(-m_predictions)[1]),
                                                "conf_top1": float(m_conf), "margin": float(m_margin),
                                                "entropy": float(m_entropy),
                                                "stylemix_seed": self.stylemix_seed, "stylemix_layer": str(layer),
                                                "alpha": float(alpha_max), "alpha_iters": iteration,
                                                "ssi": float(ssi), "l2_distance": float(l2_distance),
                                                "img_l2": float(img_l2), "m_img_l2": float(m_img_l2),
                                                "runtime_ms": m_digit_info.get("_runtime_ms",""),
                                                "timestamp": int(time.time())
                                            })
                                            break
                                        else:
                                            print("Invalid mutation - skipping")
                                    else:
                                        print(f"Misclassification to unexpected class {m_class}, expected {stylemix_cls}")

                        if _budget_exceeded():
                            print(f"[seed {self.w0_seed}] budget exceeded -> moving to next seed")
                            break

                        if found_mutation:
                            break

                        self.stylemix_seed += 1

                    if found_mutation or _budget_exceeded():
                        break

            self.w0_seed += self.step_size



def _parse_psi_list(s):
    vals = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals

def _annotate(im, text):
  
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = (max(1, int(6 * len(text))), 10)
    box_h = max(10, int(0.25 * h), th + 4)
    draw.rectangle([(0, h - box_h), (w, h)], fill=(0, 0, 0))
    tx = (w - tw) / 2
    ty = h - box_h + (box_h - th) / 2
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return im

def render_truncation_grid(
    class_idx=2,
    w0_seed=0,
    psi_list=(1.0, 0.9, 0.8, 0.7, 0.6),
    pkl_path=None,
    trunc_cutoff=4,
    annotate=False,                 
    outdir="figs/truncation",
    dataset_name="mnist",
):
   
    os.makedirs(outdir, exist_ok=True)

    state = copy.deepcopy(STYLEGAN_INIT)
    state["params"]["class_idx"] = class_idx
    state["params"]["w0_seeds"] = [[w0_seed, 1.0]]
    state["params"]["stylemix_idx"] = []
    state["params"]["mixclass_idx"] = None
    state["params"]["stylemix_seed"] = None
    state["params"]["INTERPOLATION_ALPHA"] = 1.0

    cutoff_val = trunc_cutoff if (trunc_cutoff is not None and trunc_cutoff > 0) else None

    renderer = Renderer()
    frames = []
    frame_paths = []

    for psi in psi_list:
        try:
            result = renderer.render(
                pkl=pkl_path or INIT_PKL,
                w0_seeds=state['params']['w0_seeds'],
                class_idx=state['params']['class_idx'],
                mixclass_idx=state['params']['mixclass_idx'],
                stylemix_idx=state['params']['stylemix_idx'],
                stylemix_seed=state['params']['stylemix_seed'],
                trunc_psi=float(psi),
                trunc_cutoff=cutoff_val,
                img_normalize=state['params']['img_normalize'],
                to_pil=True,
                INTERPOLATION_ALPHA=state['params']['INTERPOLATION_ALPHA'],
            )
        except Exception as e:
            _log_error(f"trunc_fig render failed @ psi={psi}", e)
            continue

        if 'image' not in result:
            print(f"[warn] render returned no image at psi={psi}")
            continue

        im = result['image'].copy()
    
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        if annotate:
            tag = f"ψ={psi:.2f}" if float(psi) != 1.0 else "base (ψ=1.00)"
            im = _annotate(im, tag)

        frame_name = f"{dataset_name}_c{class_idx}_seed{w0_seed}_psi{str(psi).replace('.','')}.png"
        frame_fp = os.path.join(outdir, frame_name)
        im.save(frame_fp)
        frame_paths.append(frame_fp)
        frames.append(im)

    if not frames:
        print("[warn] no frames rendered; aborting contact sheet")
        return

    w, h = frames[0].size
    strip = Image.new("RGB", (w * len(frames), h))
    x = 0
    for im in frames:
        strip.paste(im, (x, 0))
        x += w

    strip_name = f"{dataset_name}_c{class_idx}_seed{w0_seed}_trunc_strip.png"
    strip_fp = os.path.join(outdir, strip_name)
    strip.save(strip_fp)

    print(f"[trunc_fig] Saved strip: {strip_fp}")
    for p in frame_paths:
        print(f"[trunc_fig] Saved frame: {p}")



def run_mimicry(class_idx, w0_seed=0, step_size=1, forced_trunc_psi=None, no_adaptive=False,
                topk_targets=ENV_TOPK_TARGETS, single_layer_only=ENV_SINGLE_LAYER_ONLY,
                max_seconds_per_seed=ENV_MAX_SECONDS_PER_SEED, max_renders_per_seed=ENV_MAX_RENDERS_PER_SEED,
                truncation_mode="adaptive", psi_sweep="1.0,0.95,0.90,0.85"):
    """
    truncation_mode:
      - 'none'     -> psi fixed at 1.0, no adaptation
      - 'fixed'    -> run separate passes for each psi in psi_sweep, no adaptation
      - 'adaptive' -> start at 1.0; if baseline not accepted, adapt using built-in grid
    """
    if truncation_mode == "none":
        m = mimicry(
            class_idx=class_idx, w0_seed=w0_seed, step_size=step_size,
            forced_trunc_psi=1.0, no_adaptive=True,
            topk_targets=topk_targets, single_layer_only=single_layer_only,
            max_seconds_per_seed=max_seconds_per_seed, max_renders_per_seed=max_renders_per_seed
        )
        m.search()

    elif truncation_mode == "fixed":
        sweep = [float(x.strip()) for x in psi_sweep.split(",") if x.strip()]
        for psi in sweep:
            m = mimicry(
                class_idx=class_idx, w0_seed=w0_seed, step_size=step_size,
                forced_trunc_psi=psi, no_adaptive=True,
                topk_targets=topk_targets, single_layer_only=single_layer_only,
                max_seconds_per_seed=max_seconds_per_seed, max_renders_per_seed=max_renders_per_seed
            )
            m.search()

    else:  
        m = mimicry(
            class_idx=class_idx, w0_seed=w0_seed, step_size=step_size,
            forced_trunc_psi=None, no_adaptive=False,
            topk_targets=topk_targets, single_layer_only=single_layer_only,
            max_seconds_per_seed=max_seconds_per_seed,
            max_renders_per_seed=max_renders_per_seed
        )
        m.search()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--class_idx", type=int, default=2)
    p.add_argument("--w0_seed", type=int, default=0)
    p.add_argument("--step_size", type=int, default=1)
    p.add_argument("--psi", type=float, default=None, help="Force truncation psi (e.g., 1.0, 0.95).")
    p.add_argument("--no_adaptive", action="store_true", help="If psi is set, skip fallback grid for fair comparison.")
 
    p.add_argument("--topk_targets", type=int, default=ENV_TOPK_TARGETS)
    p.add_argument("--single_layer_only", type=int, default=int(ENV_SINGLE_LAYER_ONLY), help="1 only single layers; 0 include combos")
    p.add_argument("--max_seconds_per_seed", type=int, default=ENV_MAX_SECONDS_PER_SEED)
    p.add_argument("--max_renders_per_seed", type=int, default=ENV_MAX_RENDERS_PER_SEED)
    p.add_argument("--quick", action="store_true", help="Shortcut: single-layer, top2, 15s/seed, 300 renders/seed.")
 
    p.add_argument(
        "--truncation_mode",
        type=str,
        choices=["none", "fixed", "adaptive"],
        default="adaptive",
        help="none: psi=1.0 no-adapt; fixed: run a sweep of psi values without adaptation; adaptive: start at 1.0 and adapt if needed."
    )
    p.add_argument(
        "--psi_sweep",
        type=str,
        default="1.0,0.95,0.90,0.85",
        help="Comma-separated list of psi values used when --truncation_mode=fixed."
    )


    p.add_argument("--trunc_fig", action="store_true",
                   help="Render a truncation strip (base + psi list) for a single (class, seed) and exit.")
    p.add_argument("--psi_list", type=str, default="1.0,0.9,0.8,0.7,0.6",
                   help="Comma-separated psi values for the figure (first can be 1.0 for base).")
    p.add_argument("--trunc_cutoff", type=int, default=4,
                   help="Apply truncation only to first N W+ layers (0 or negative = all layers).")
    p.add_argument("--pkl", type=str, default=None,
                   help="Override generator .pkl (use this for CIFAR-10 or F-MNIST checkpoints).")
    p.add_argument("--outdir", type=str, default="figs/truncation",
                   help="Output directory for the figure and frames.")
    p.add_argument("--dataset_name", type=str, default="mnist",
                   help="Used in filenames only (mnist/cifar10/fmnist).")
    p.add_argument("--annotate", action="store_true",
                   help="If set, draw ψ labels on each frame (OFF by default).")

    args = p.parse_args()

    if args.quick:
        args.single_layer_only = 1
        args.topk_targets = 2
        args.max_seconds_per_seed = 15
        args.max_renders_per_seed = 300

   
    if args.trunc_fig:
        psi_vals = _parse_psi_list(args.psi_list)
        if not psi_vals:
            psi_vals = [1.0, 0.9, 0.8, 0.7, 0.6]
        render_truncation_grid(
            class_idx=args.class_idx,
            w0_seed=args.w0_seed,
            psi_list=psi_vals,
            pkl_path=args.pkl,
            trunc_cutoff=args.trunc_cutoff,
            annotate=args.annotate,      
            outdir=args.outdir,
            dataset_name=args.dataset_name,
        )
        sys.exit(0)


    run_mimicry(
        class_idx=args.class_idx, w0_seed=args.w0_seed, step_size=args.step_size,
        forced_trunc_psi=args.psi, no_adaptive=args.no_adaptive,
        topk_targets=args.topk_targets, single_layer_only=bool(args.single_layer_only),
        max_seconds_per_seed=args.max_seconds_per_seed, max_renders_per_seed=args.max_renders_per_seed,
        truncation_mode=args.truncation_mode, psi_sweep=args.psi_sweep
    )
