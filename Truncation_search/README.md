# truncation-search method

Fault‑Revealing Input Generation For Deep Learning-Based System Using Only Latent‑Space Truncation.

## Installation

```bash
conda env create -f environment.yml
conda activate Trunc
```

## Getting started

1. Download pre-trained networks `*.pkl` files from [Huggingface](https://huggingface.co/awafa/cSG2) and put them under `./checkpoints/checkpoints`.

2) Adjust paths in config.py files.

3) Run: cd <dataset>/python search.py
   

## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3) and [PTI](https://github.com/tianhaoxie/DragGAN_PTI/tree/27a9821085ce4d9b788aaf4bbb52b9b982b25bcd?tab=readme-ov-file)
