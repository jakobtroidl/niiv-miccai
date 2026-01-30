from setuptools import setup, find_packages

setup(
    name='niiv',
    version='0.0.5',
    packages=find_packages(),
    install_requires=[
        'torch',
        'DISTS_pytorch',
        'lpips',
        'numpy',
        'pytorch-ignite',
        'tqdm',
        'pytorch_msssim'
    ],
    # Other metadata
    author='Jakob Troidl',
    author_email='jakob_troidl@fas.harvard.edu',
    description='Self-Supervised Neural Implicit Isotropic Volume Reconstruction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important for a proper display on PyPI
    url='https://github.com/jakobtroidl/niiv-miccai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
