# niiv 
Code for the paper _[niiv: Interactive Self-supervised Neural Implicit Isotropic Volume Reconstruction](https://jakobtroidl.github.io/assets/pdf/niiv_preprint.pdf)_ published at the MICCAI Workshop on Efficient Medical AI (EMA), 2025. 

[J. Troidl](https://jakobtroidl.github.io/), [Y. Liang](https://lynl7130.github.io/), [J. Beyer](https://johanna-b.github.io/), [M. Tavakoli](https://www.janelia.org/people/mojtaba-tavakoli), [J. Danzl](https://danzl-lab.pages.ist.ac.at/), [M. Hadwiger](https://www.kaust.edu.sa/en/study/faculty/markus-hadwiger), [H. Pfister](https://vcg.seas.harvard.edu/people), [J. Tompkin](https://jamestompkin.com/)



## Abstract
Three-dimensional (3D) microscopy data often is anisotropic with significantly lower resolution (up to 8x) along the z axis than along the xy axes. Computationally generating plausible isotropic resolution from anisotropic imaging data would benefit the visual analysis of large-scale volumes. This paper proposes niiv, a self-supervised method for isotropic reconstruction of 3D microscopy data that can quickly produce images at arbitrary output resolutions. The representation embeds a learned latent code within a neural field that describes the implicit higher-resolution isotropic image region. We use a novel attention-guided latent interpolation approach, which allows flexible information exchange over a local latent neighborhood. Under isotropic volume assumptions, we self-supervise this representation on low-/high-resolution lateral image pairs to reconstruct an isotropic volume from low-resolution axial images. We evaluate our method on simulated and real anisotropic electron (EM) and light microscopy (LM) data. Compared to a state-of-the-art diffusion-based method, niiv shows improved reconstruction quality (+1dB PSNR) and is over three orders of magnitude faster (1,000x) to infer. Specifically, niiv reconstructs a 128^3 voxel volume in 2/10th of a second, renderable at varying (continuous) high resolutions for display.

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA 11.7 or higher (for GPU support)
- Conda package manager

### Setup

1. Create and activate a new conda environment:
```bash
conda create -n niiv python=3.9
conda activate niiv
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

## Usage

### Training

Train a model on your dataset using the training script:

```bash
python experiments/train_images.py \
    --config ./config/config_cvr_s.json \
    --dataset /path/to/your/dataset/info.json \
    --experiment_name my_experiment \
    --num_epochs 1500 \
    --lr 0.00005 \
    --batch_size 70
```

### Evaluation

Evaluate a trained model:

```bash
python experiments/eval_seq_simplified.py \
    --config ./config/config_cvr_s.json \
    --dataset /path/to/your/dataset/info.json \
    --experiment_name my_experiment \
    --checkpoint /path/to/checkpoint.pth
```

### Configuration

Configuration files in the `config/` directory control model architecture and training hyperparameters:
- `config_cvr_s.json` - Standard model configuration
- `config_cvr_tiny.json` - Smaller model for faster training
- `config_siren.json` - Alternative SIREN-based configuration

## Dataset Format

The dataset should be organized with the following structure:

```
data/
└── your_dataset/
    ├── info.json
    ├── train/
    │   ├── volume_0.npy
    │   ├── volume_1.npy
    │   └── ...
    └── test/
        ├── volume_0.npy
        └── ...
```

### info.json Format

The `info.json` file should contain dataset metadata:

```json
{
  "train_path": "data/your_dataset/train",
  "test_path": "data/your_dataset/test",
  "resolution": [8, 1, 1],
  "image_size": 128
}
```

### Volume Format

Each `.npy` file should contain a 3D numpy array of shape `(D, H, W)` with dtype `uint8` (0-255 range).

## Project Structure

```
niiv-miccai/
├── config/               # Model configuration files
├── dataio.py            # Dataset loading utilities
├── experiments/         # Training and evaluation scripts
│   ├── train_images.py
│   ├── eval_seq_simplified.py
│   └── create_dataset.py
├── examples/            # Example notebooks and analysis
│   └── analysis/
├── niiv/               # Core model implementation
│   ├── models.py       # Main NIIV model
│   ├── feature_grid.py # Feature grid processing
│   ├── encoders/       # Encoder architectures
│   ├── decoder/        # Decoder architectures
│   └── util/          # Utility functions
├── scripts/           # Shell scripts for training/testing
├── training.py        # Training loop implementation
└── visualization/     # Visualization tools
```

## Citation

```bibtex
@inproceedings{troidl2025niiv,
  title={niiv: Interactive Self-supervised Neural Implicit Isotropic Volume Reconstruction},
  author={Troidl, Jakob and Liang, Yiqing and Beyer, Johanna and Tavakoli, Mojtaba and Danzl, Johann and Hadwiger, Markus and Pfister, Hanspeter and Tompkin, James},
  booktitle={International Workshop on Efficient Medical Artificial Intelligence},
  pages={257--267},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon code from the following repositories:
- [NVP](https://github.com/subin-kim-cv/NVP) - Neural Volume Prior
- [LIIF](https://github.com/yinboc/liif) - Learning Implicit Image Functions
- [VINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution) - Video INR

We thank the authors for making their code publicly available. Claude Code supported the GitHub release and polished the repository for release. 
