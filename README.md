## VQ-VAE with FSQ on MNIST

This project trains a VQ-VAE model on MNIST using a Finite Scalar Quantization (FSQ) quantizer.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Train

```bash
python main.py --epochs 10 --batch-size 128 --levels 8,8,8,8
```

Outputs are written to `outputs/<timestamp>/` with checkpoints and reconstruction grids.

### Useful flags

- `--levels`: Comma-separated per-dimension levels for FSQ (e.g. `8,8,8,8`).
- `--beta`: Commitment loss weight.
- `--device`: `auto`, `cpu`, or `cuda`.
- `--output-dir`: Where to save checkpoints and reconstructions.
