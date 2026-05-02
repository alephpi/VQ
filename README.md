## VQ-VAE with FSQ/BSQ on MNIST

A simple tutorial of VQ-VAE model on MNIST with Finite Scalar Quantization (FSQ) / Binary Spherical Quantization (BSQ).

### Setup

```bash
uv sync
source .venv/bin/activate
```

### Train

```bash
python main.py --epochs 40 --batch-size 128 --quantizer fsq --levels 5,5,5
python main.py --epochs 40 --batch-size 128 --quantizer bsq --codebook-size 1024
```

Outputs are written to `outputs/<timestamp>/` with checkpoints and reconstruction grids.

### Some data
| Quantizer | Setting | recon_loss | codebook usage |
| --- | --- | --- | --- |
| FSQ | 5,5,5 | 0.05 | 0.7 |
| FSQ | 8,8,8 | 0.04 | 0.4 |
| BSQ | 128 (2^7) | 0.05 | 0.9 |
| BSQ | 1024 (2^10) | 0.04 | 0.5 |