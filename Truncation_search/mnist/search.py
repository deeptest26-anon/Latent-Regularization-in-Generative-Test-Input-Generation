import os
import copy
import json
import numpy as np
import sys


import torch
try:
    import lpips
    _LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    _LPIPS_AVAILABLE = False
    print("[warn] lpips not installed; LPIPS validation will be skipped. Try: pip install lpips")

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

def compute_lpips_distance(img1_np, img2_np):
  
    if not _LPIPS_AVAILABLE:
        return None
    model, device = _get_lpips_model()
    if model is None:
        return None

 
    x = torch.from_numpy(img1_np).float() / 255.0
    y = torch.from_numpy(img2_np).float() / 255.0
    if x.ndim == 2:  
        x = x.unsqueeze(0)  
    if y.ndim == 2:
        y = y.unsqueeze(0)


    x = x.repeat(3, 1, 1)  
    y = y.repeat(3, 1, 1)

 
    x = x.unsqueeze(0)  
    y = y.unsqueeze(0)

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

# -----------------------------------------------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, INIT_PKL, FRONTIER_PAIRS
from predictor import Predictor

def convert_types(o):
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

class mimicry:
    def __init__(self, class_idx=None, w0_seed=0, search_limit=SEARCH_LIMIT, step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.search_limit = search_limit
        self.step_size = step_size
        self.state['renderer'] = Renderer()

    def render_state(self, state=None):
        if state is None:
            state = self.state
        result = state['renderer'].render(
            pkl=INIT_PKL,
            w0_seeds=state['params']['w0_seeds'],
            class_idx=state['params']['class_idx'],
            trunc_psi=state['params']['trunc_psi'],
            trunc_cutoff=state['params']['trunc_cutoff'],
            img_normalize=state['params']['img_normalize'],
            to_pil=state['params']['to_pil'],
        )
        info = copy.deepcopy(state['params'])
        return result, info

    def search(self):


        root = os.path.join(FRONTIER_PAIRS, str(self.class_idx))
        frontier_seed_count = 0

        while frontier_seed_count < self.search_limit:
            state = self.state

     
            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]

            
            state['params']['trunc_psi'] = 1.0
            state['params']['trunc_cutoff'] = None

            
            digit, digit_info = self.render_state()
            if 'image' not in digit:
                print(f"Render failed with error: {digit.get('error', 'Unknown error')}")
                self.w0_seed += self.step_size
                continue

            base_label = digit_info["class_idx"]
            base_image = digit['image']
     
            base_image = base_image.crop((2, 2, base_image.width - 2, base_image.height - 2))
            base_array = np.array(base_image)

      
            base_accepted, base_confidence, base_predictions = Predictor().predict_datapoint(
                np.reshape(base_array, (-1, 28, 28, 1)),
                base_label
            )
            base_pred = base_label if base_accepted else int(np.argmax(base_predictions))
            print(f"Base image generated with predicted class {base_pred} for seed {self.w0_seed}")

      
            if base_pred != self.class_idx:
                print(f"Base image prediction {base_pred} does not match the expected class {self.class_idx}. Skipping seed {self.w0_seed}.")
                self.w0_seed += self.step_size
                continue


            fault_found = False
            truncation_values = np.arange(0.9, 0.5, -0.05)
            for trunc_psi in truncation_values:
                state['params']['trunc_psi'] = float(trunc_psi)
                fault_digit, fault_digit_info = self.render_state()

                if 'image' not in fault_digit:
                    print(f"Render failed with error: {fault_digit.get('error', 'Unknown error')} at trunc_psi={trunc_psi}")
                    continue

                fault_image = fault_digit['image']
                fault_image = fault_image.crop((2, 2, fault_image.width - 2, fault_image.height - 2))
                fault_array = np.array(fault_image)

                fault_accepted, fault_confidence, fault_predictions = Predictor().predict_datapoint(
                    np.reshape(fault_array, (-1, 28, 28, 1)),
                    base_label
                )
                fault_pred = base_label if fault_accepted else int(np.argmax(fault_predictions))
                print(f"trunc_psi {trunc_psi} produced predicted class {fault_pred} (base was {base_pred})")

                
                if fault_pred != base_pred:
                    path = os.path.join(root, str(self.w0_seed))
                    os.makedirs(path, exist_ok=True)

  
                    base_img_path = os.path.join(path, "base.png")
                    base_image.save(base_img_path)
                    print(f"Base image saved at {base_img_path} with predicted class {base_pred}")

     
                    meta_base = copy.deepcopy(digit_info)
                    meta_base["accepted"] = bool(base_accepted)
                    meta_base["exp-confidence"] = float(base_confidence)
                    meta_base["predictions"] = base_predictions.tolist() if hasattr(base_predictions, 'tolist') else list(base_predictions)
                    meta_base["trunc_psi"] = 1.0
                    meta_base["lpips"] = 0.0 
                    with open(os.path.join(path, "base.json"), 'w') as f:
                        json.dump(meta_base, f, sort_keys=True, indent=4, default=convert_types)

         
                    fault_img_name = f"fault_trunc_{float(trunc_psi):.2f}.png"
                    fault_img_path = os.path.join(path, fault_img_name)
                    fault_image.save(fault_img_path)
                    print(f"Fault revealing image saved at {fault_img_path} with trunc_psi {trunc_psi}")

                    lpips_value = compute_lpips_distance(base_array, fault_array)
                    if lpips_value is not None:
                        print(f"LPIPS(base, fault) = {lpips_value:.6f}")
                    else:
                        print("LPIPS unavailable; skipping perceptual distance computation.")

                    meta_fault = copy.deepcopy(fault_digit_info)
                    meta_fault["accepted"] = bool(fault_accepted)
                    meta_fault["exp-confidence"] = float(fault_confidence)
                    meta_fault["predictions"] = fault_predictions.tolist() if hasattr(fault_predictions, 'tolist') else list(fault_predictions)
                    meta_fault["trunc_psi"] = float(trunc_psi)
                    meta_fault["fault_pred"] = int(fault_pred)
                    meta_fault["lpips"] = lpips_value  
                    with open(os.path.join(path, f"fault_trunc_{float(trunc_psi):.2f}.json"), 'w') as f:
                        json.dump(meta_fault, f, sort_keys=True, indent=4, default=convert_types)

                    fault_found = True
                    break

            if not fault_found:
                print("No fault revealing image found for this seed.")

            frontier_seed_count += 1
            self.w0_seed += self.step_size

def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry_instance = mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size)
    mimicry_instance.search()

if __name__ == "__main__":
    run_mimicry(class_idx=2)
