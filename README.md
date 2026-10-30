# GCATransformer

Official implementation of **Group Co-attention Transformer for Image Co-segmentation**, published in *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*.

[[Paper]](https://ieeexplore.ieee.org/document/11649503)

## 1. Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/<username>/GCATransformer.git
cd GCATransformer

conda env create -f environment.yml
conda activate <env_name>
```

## 2. Dataset Preparation

Download the datasets and organize them under the `data/` directory as follows:

```text
data/
├── Internet/
│   └── test/
│       └── Car/
│           ├── img/
│           │   └── 0002.png
│           └── lb/
│               └── 0002.png
├── MSRC/
│   └── test/
│       └── bird/
│           ├── img/
│           └── lb/
├── PASCAL/
│   └── test/
│       └── bicycle/
│           ├── img/
│           └── lb/
└── iCoSeg/
    └── test/
        └── bear2/
            ├── img/
            └── lb/
```

Each category contains an `img/` folder for input images and an `lb/` folder for the corresponding ground-truth masks.

## 3. Testing

Download the weights used in our paper:

[[Weights]](https://huggingface.co/jiunfeng845bd/GCATransformer/resolve/main/GCATransformer_tcsvt2026.sav)

Run evaluation using:

```bash
python test.py --weights <path_to_weights>
```

## 4. Training

To train GCATransformer from scratch, run:

```bash
python train.py
```

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{chang2026group,
  author  = {Chang, Jiun-Feng and Lin, Jason and Huang, Chun-Rong},
  title   = {Group Co-attention Transformer for Image Co-segmentation},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  volume  = {36},
  number  = {11},
  pages   = {17628-17640},
  doi     = {10.1109/TCSVT.2026.3722576}
}
```

## License

This project is released under the Apache License 2.0.
