# Code for "Maximum a posteriori natural scene reconstruction from retinal ganglion cells with deep denoiser priors"

## Dependencies

1. requires Python >= 3.8
2. numpy, scipy, matplotlib, pytorch, h5py, tqdm, shapely
3. To run the provided notebook, a working Jupyter installation is needed as well

An NVIDIA GPU is strongly suggested to run the sample code.

## What's in here?

This repository contains code for reconstructing example images for retinal ganglion cell spikes
for all of the (approximate) MAP methods presented in the paper (MAP-GLM-dCNN, MAP-LNP-dCNN, MAP-GLM-1F) as well as
all of the benchmark neural network regression methods. The repository also includes example scripts demonstrating
how the RGC encoding models were fit. **Because only a toy subset of the test partition of the  experimental dataset is provided, these example
model fitting scripts are solely for demonstration purposes only, and should not be used for image reconstruction.**

The toy datasets as well as fitted model parameters for all of the encoding models and neural networks are provided 
on Zenodo.

Dataset and fitted models:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6582341.svg)](https://doi.org/10.5281/zenodo.6582341)

A Jupyter notebook (reconstructions.ipynb) explaining the dataset and illustrating how to run the code is also provided.

## Getting the dataset and models

## Generating example reconstructions

#### MAP-GLM-dCNN reconstructions (GLM with denoiser prior)

To generate MAP-GLM-dCNN reconstructions using random initialization with noise standard deviation 1e-2,
and save the outputs in the path `OUTPUT_RECONSTRUCTIONS/map_glm_dCNN.p`, run the command 

```shell
python generate_MAP_GLM_dCNN_recons.py OUTPUT_RECONSTRUCTIONS/map_glm_dCNN.p -gpu -n 1e-2
```

To generate MAP-GLM-dCNN reconstructions using initialization with linear reconstruction (the initialization method
shown the main text of the paper), run

```shell
python generate_MAP_GLM_dCNN_recons.py OUTPUT_RECONSTRUCTIONS/map_glm_dCNN.p -gpu -l
```

There is an additional `-b [BATCH-SIZE]` flag that allows the user to specify the batch size for doing the reconstructions
in parallel. This does not affect the output images. Larger batch sizes run faster because of improved parallelism, and smaller batch sizes may be necessary to fit 
on smaller GPUs.

#### MAP-LNP-dCNN reconstructions (LNP with denoiser prior)

MAP-GLM-dCNN reconstruction generation has the same command line arguments as MAP-GLM-dCNN. To generate MAP-GLM-dCNN reconstructions using random initialization with noise standard deviation 1e-2,
and save the outputs in the path `OUTPUT_RECONSTRUCTIONS/map_lnp_dCNN.p`, run the command 

```shell
python generate_MAP_LNP_dCNN_recons.py OUTPUT_RECONSTRUCTIONS/map_lnp_dCNN.p -gpu -n 1e-2
```

To generate MAP-LNP-dCNN reconstructions using initialization with linear reconstruction, run

```shell
python generate_MAP_LNP_dCNN_recons.py OUTPUT_RECONSTRUCTIONS/map_lnp_dCNN.p -gpu -l
```

There is an additional `-b [BATCH-SIZE]` flag that allows the user to specify the batch size for doing the reconstructions
in parallel. Larger batch sizes run faster because of improved parallelism, and smaller batch sizes may be necessary to fit 
on smaller GPUs.

#### MAP-GLM-1F reconstructions (LNP with 1/F prior)

To generate MAP-GLM-1F reconstructions, run the command

```shell
python generate_MAP_GLM_1F_recons.py OUTPUT_RECONSTRUCTIONS/map_glm_1f.p -gpu
```

There is an additional `-b [BATCH-SIZE]` flag that allows the user to specify the batch size for doing the reconstructions
in parallel.

#### L-CAE and linear reconstruction benchmarks

To generate L-CAE [1] and linear [2, 3] benchmark reconstructions, run 

```shell
python generate_lcae_recons.py OUTPUT_RECONSTRUCTIONS/lcae.p -gpu
```

There is an additional `-b [BATCH-SIZE]` flag that allows the user to specify the batch size for doing the reconstructions
in parallel.

#### Kim et al. neural network regression benchmark

To generate the Kim _et al._ [4] benchmark reconstructions, run

```shell
python generate_kim_recons.py OUTPUT_RECONSTRUCTIONS/kim.p -gpu
```

There is an additional `-b [BATCH-SIZE]` flag that allows the user to specify the batch size for doing the reconstructions
in parallel.

####

## Fitting models to the toy datasets

> :warning: **The provided toy dataset is insufficient to produce high-quality model fits; 
> this code provided here is meant only to demonstrate how the model fitting works.**

#### Fitting the GLMs

The code to fit GLMs [5] to the provided toy dataset using the same hyperparameters as in the published work is provided in
`demo_glm_fit.py`. Because fitting the GLMs is relatively expensive computationally, the GLMs are fit separately for each of the major
RGC cell types (ON parasol, OFF parasol, ON midget, OFF midget), and the cell type must be specified at the command line.

For example, to fit GLMs to all of the ON parasols, and save the model fits to `OUTPUT_ENCODING_MODELS/on_parasol_glm.p`,
run the command 

```shell
python demo_glm_fit.py 'ON parasol' OUTPUT_ENCODING_MODELS/on_parasol_glm.p -gpu
```

#### Fitting the LNPs

The code to fit LNP models to the provided toy dataset using the same hyperparameters as in the published work are provided
in `demo_fit_scaled_LNP.py` and `demo_fit_full_LNP.py`. The LNP fitting requires two steps:

1. Fitting an initial scaled filter and bias from the spatial filters calculated via reverse correlation
2. Fitting the filter and bias using the filter from the previous step as a prior.

Step 1 can be run with the command

```shell
python demo_fit_scaled_LNP.py OUTPUT_ENCODING_MODELS/scaled_lnp.p -gpu
```

and step 2 can be run with the command 

```shell
python demo_fit_scaled_LNP.py OUTPUT_ENCODING_MODELS/scaled_lnp.p OUTPUT_ENCODING_MODELS/full_lnp.p -gpu
```

## References
1. N. Parthasarathy, E. Batty, W. Falcon, T. Rutten, M. Rajpal, E. Chichilnisky, and L. Paninski, “Neural Networks for Efficient Bayesian Decoding of Natural Images from Retinal Neurons,”
conference, Neuroscience, June 2017.
2. D. K. Warland, P. Reinagel, and M. Meister, “Decoding Visual Information From a Population of Retinal Ganglion Cells,” Journal of Neurophysiology, vol. 78, pp. 2336–2350, Nov. 1997.
3. N. Brackbill, C. Rhoades, A. Kling, N. P. Shah, A. Sher, A. M. Litke, and E. Chichilnisky, “Reconstruction of natural images from responses of primate retinal ganglion cells,” eLife, vol. 9,
p. e58516, Nov. 2020.
4. Y.J. Kim, N. Brackbill, E. Batty, J. Lee, C. Mitelut, W. Tong, E.J. Chichilnisky, and L. Paninski, “Nonlinear Decoding of Natural Images From Large-Scale Primate Retinal Ganglion Recordings,” Neural Computation, vol. 33, pp. 1719–1750, June 2021.
5. J.W. Pillow, J. Shlens, L. Paninski, A. Sher, A.M. Litke, E.J. Chichilnisky, and E.P. Simoncelli, “Spatio-temporal correlations and visual signalling in a complete neuronal population,” _Nature_, vol. 454, pp. 995-999, Aug. 2008.
