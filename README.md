# phase_learn
Python code (tested on 2.7) for distribution regression using features based on empirical phase functions.

The method is described in:
H. C. L. Law, C. Yau, and D. Sejdinovic, __Testing and Learning on Distributions with Symmetric Noise Invariance__, in _Advances in Neural Information Processing Systems (NIPS)_, 2017. [arxiv](https://arxiv.org/abs/1703.07596)

The phase features with ridge regression are incorporated as a part of the [kerpy package](https://github.com/oxmlcs/kerpy).

### Notebooks
* The notebook tutorial_toy.ipynb demonstrates the use of the neural network implementations for learning Fourier features on a toy classification problem. 
* The notebook Phase_Fourier_ridge_tutorial.ipynb demonstrates the use of the phase ridge regression (using kerpy) with the MISR1 aerosol dataset.
* Notebook Phase_Neural_Network_Aerosol.ipynb demonstrates neural network implementations for learning both phase and Fourier features on the MISR1 aerosol dataset. 

The MISR1 dataset in MISR1.mat is originally from http://www.dabi.temple.edu/~vucetic/MIR.html.
