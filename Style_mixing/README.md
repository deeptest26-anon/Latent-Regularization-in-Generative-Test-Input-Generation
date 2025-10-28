# style-mixing method

## Installation

```bash
conda env create -f environment.yml
conda activate Trunc
```

## Getting started

1. Download pre-trained networks `*.pkl` files from [Huggingface](https://huggingface.co/awafa/cSG2) and put them under `./checkpoints/checkpoints`.

2. Select one of the following datasets and go to the corresponding directory:
  - MNIST: `./mnist` 
  - FashionMNIST: `./f-mnist` 
  - CIFAR-10: `./CIFAR-10` 

3. Adjust the default config in `./<DATASET>/config.py`.
   - MNIST: Change paths in config.py
   - FashionMNIST: Change paths in config.py and in Model1_fmnist.py
   - CIFAR-10: Change paths in config.py
   
5. Run the file `./<DATASET>/search.py` to generate frontier pairs.


No Truncation Run: python search.py --truncation_mode none --class_idx 0 --w0_seed 0

Fixed Truncation Run: python search.py --truncation_mode fixed --psi_sweep "0.8" --class_idx 0 --w0_seed 0

Adaptive: python search.py --truncation_mode adaptive --class_idx 0 --w0_seed 0




## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3) and [PTI](https://github.com/tianhaoxie/DragGAN_PTI/tree/27a9821085ce4d9b788aaf4bbb52b9b982b25bcd?tab=readme-ov-file)
