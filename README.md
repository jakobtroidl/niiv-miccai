# niiv 
Code for the paper _[niiv: Interactive Self-supervised Neural Implicit Isotropic Volume Reconstruction](https://jakobtroidl.github.io/assets/pdf/niiv_preprint.pdf)_ published at the MICCAI Workshop on Efficient Medical AI (EMA), 2025. 

[J. Troidl](https://jakobtroidl.github.io/), [Y. Liang](https://lynl7130.github.io/), [J. Beyer](https://johanna-b.github.io/), [M. Tavakoli](https://www.janelia.org/people/mojtaba-tavakoli), [J. Danzl](https://danzl-lab.pages.ist.ac.at/), [M. Hadwiger](https://www.kaust.edu.sa/en/study/faculty/markus-hadwiger), [H. Pfister](https://vcg.seas.harvard.edu/people), [J. Tompkin](https://jamestompkin.com/)



## Abstract
Three-dimensional (3D) microscopy data often is anisotropic with significantly lower resolution (up to 8x) along the z axis than along the xy axes. Computationally generating plausible isotropic resolution from anisotropic imaging data would benefit the visual analysis of large-scale volumes. This paper proposes niiv, a self-supervised method for isotropic reconstruction of 3D microscopy data that can quickly produce images at arbitrary output resolutions. The representation embeds a learned latent code within a neural field that describes the implicit higher-resolution isotropic image region. We use a novel attention-guided latent interpolation approach, which allows flexible information exchange over a local latent neighborhood. Under isotropic volume assumptions, we self-supervise this representation on low-/high-resolution lateral image pairs to reconstruct an isotropic volume from low-resolution axial images. We evaluate our method on simulated and real anisotropic electron (EM) and light microscopy (LM) data. Compared to a state-of-the-art diffusion-based method, niiv shows improved reconstruction quality (+1dB PSNR) and is over three orders of magnitude faster (1,000x) to infer. Specifically, niiv reconstructs a 128^3 voxel volume in 2/10th of a second, renderable at varying (continuous) high resolutions for display.

## Getting Started

```
conda create -n niiv python=3.9
conda activate niiv
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
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

## References
We used the code from following repositories: [NVP](https://github.com/subin-kim-cv/NVP), [LIIF](https://github.com/yinboc/liif), [VINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution).
