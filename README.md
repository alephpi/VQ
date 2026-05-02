## VQ-VAE with FSQ on MNIST

This project trains a VQ-VAE model on MNIST using a Finite Scalar Quantization (FSQ) quantizer and a Binary Spherical Quantization (BSQ).

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
FSQ 5,5,5: recon_loss = 0.05, codebook usage = 0.7
FSQ 8,8,8: recon_loss = 0.04, codebook usage = 0.4
BSQ 128(2^7): recon_loss = 0.05 , codebook usage = 0.9
BSQ 1024(2^10): recon_loss = 0.04, codebook usage = 0.5