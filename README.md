# phase_learn
Python code (tested on 2.7) for phase features in ridge regression and phase neural network for distributional regression. The phase features with ridge regression can be incorporated as part of the kerpy package from https://github.com/oxmlcs/kerpy.

The method is described in:

Testing and Learning on Distributions with Symmetric Noise Invariance (https://arxiv.org/abs/1703.07596)

The notebook Phase_Fourier_ridge_tutorial.ipynb demonstrate the use of the phase ridge regression (using kerpy package). Notebook Phase_Neural_Network_Aerosol.ipynb demonstrates phase and fourier neural network on distributional regression with the MISR1 aerosol dataset. The notebook tutorial_toy.ipynb demonstrates the use of the fourier neural network on a toy classification problem. 

The MISR1 dataset can be found in MISR1.mat here or at http://www.dabi.temple.edu/~vucetic/MIR.html.
Code will be updated and tidied up, with test units.
