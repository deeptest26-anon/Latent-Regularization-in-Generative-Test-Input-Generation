import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dnnlib
import torch 

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# StyleGAN2 model checkpoint
INIT_PKL = 'Path'
# Model used for prediction
MODEL = 'Path'
num_classes = 10

# Path to save the generated frontier pairs
FRONTIER_PAIRS = 'mnist/eval'

# List of layers to perform stylemix (as provided)
STYLEMIX_LAYERS = [[7], [6], [5], [4], [3], [5,6], [3,4], [3,4,5,6]]

# Number of frontier pair samples to generate
SEARCH_LIMIT = 100

# Max number of stylemix seeds
STYLEMIX_SEED_LIMIT = 100

# Loosened thresholds
SSIM_THRESHOLD = 0.95
L2_RANGE = 0.2

# Safer truncation defaults (apply cutoff from CLI during runs)
TRUNC_PSI = 1.0
TRUNC_CUTOFF = None

STYLEGAN_INIT = {
    "generator_params": dnnlib.EasyDict(),
    "params": {
        "w0_seeds": [[0, 1]],
        "w_load": None,
        "class_idx": None,
        "mixclass_idx": None,
        "stylemix_idx": [],
        "patch_idxs": None,
        "stylemix_seed": None,
        "trunc_psi": TRUNC_PSI,
        "trunc_cutoff": TRUNC_CUTOFF,
        "random_seed": 0,
        "noise_mode": 'random',
        "force_fp32": False,
        "layer_name": None,
        "sel_channels": 3,
        "base_channel": 0,
        "img_scale_db": 0,
      
        "img_normalize": False,
        "to_pil": True,
        "input_transform": None,
        "untransform": False
    },
    "device": DEVICE,
    "renderer": None,
    'pretrained_weight': INIT_PKL
}

#################################################

DJ_DEBUG = 1

# GA Setup
POPSIZE = 100

STOP_CONDITION = "iter"
# STOP_CONDITION = "time"

NGEN = 10
RUNTIME = 3600
STEPSIZE = 10

# Mutation Hyperparameters
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1

# Archive configuration
ARCHIVE_THRESHOLD = 4.0

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28

# INITIALPOP = 'seeded'
INITIALPOP = 'random'

# GENERATE_ONE_ONLY = False
GENERATE_ONE_ONLY = True

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'mnist/original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'
