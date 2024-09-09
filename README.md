# HYBRID SIAC - DATA-DRIVEN FILTER

This code develops and applies a hybrid Smoothness-Increasing Accuracy-Conserving (SIAC) -- data-driven filter to discontinuous Galerkin (DG) data with discontinuities as presented in "A hybrid SIAC - data-driven post-processing filter for discontinuities in solutions to numerical PDEs" by Soraya Terrab, Samy Wu Fung, and Jennifer K. Ryan (https://arxiv.org/abs/2408.05193). 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. Clone the following repositories, which we use for the SIAC filter and the Riemann solver, respectively: 

        1. SIACPythonCode: https://github.com/jennkryan/SIACPythonCode.git
    
        2. riemann_book: https://github.com/clawpack/riemann_book.git 
            Riemann Problems and Jupyter Solutions
                Theory and Approximate Solvers for Hyperbolic PDEs
            by David I. Ketcheson, Randall J. LeVeque, and Mauricio J. del Razo
            SIAM, 2020.   ISBN: 978-1-611976-20-5
            ebook: DOI 10.1137/1.9781611976212

2. Install the packages listed in requirements.txt:
        $ pip install -r requirements.txt

## Usage

Run the Python notebooks to execute the hybrid SIAC - data-driven framework as described below:
1. generate_training_data.ipynb: generates synthetic top-hat training data about discontinuities for the CNN filter, including the CFD discontinuity data for the validation and test datasets 
2. train_nn_filter.ipynb: trains a CNN filter that learns from the top-hat training data with CFD cross-validation
3. filter_CFD_hybrid.ipynb: constructs the hybrid filter approximation for the CFD test data



