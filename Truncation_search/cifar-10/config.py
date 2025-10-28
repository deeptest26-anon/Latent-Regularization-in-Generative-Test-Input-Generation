import dnnlib
import torch

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# StyleGAN2 model checkpoint
INIT_PKL = 'Path'
# Model used for prediction
MODEL = 'Path'
num_classes = 10

# Path to save the generated frontier pairs
FRONTIER_PAIRS = 'cifar-10/eval'
# Number of frontier pair samples to generate
SEARCH_LIMIT = 1000
# Max number of stylemix seeds
STYLEMIX_SEED_LIMIT = 100

STYLEGAN_INIT = {
    "generator_params": dnnlib.EasyDict(),
    "params": {
        "w0_seeds": [[0, 1]],
        "w_load": None,
        "class_idx": None,
        "patch_idxs": None,
        "TRUNC_PSI": None,
        "trunc_cutoff": None,
        "random_seed": 0,
        "noise_mode": 'random',
        "force_fp32": False,
        "layer_name": None,
        "sel_channels": 1,
        "base_channel": 0,
        "img_scale_db": 0,
        "img_normalize": True,
        "to_pil": True,
        "input_transform": None,
        "untransform": False
    },
    "device": DEVICE,
    "renderer": None,
    'pretrained_weight': INIT_PKL
}
