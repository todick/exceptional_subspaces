# Finding Exceptional Configuration Subspaces: An Approach Based on Subgroup Discovery

This repository contains supplementary material for the paper "Finding Exceptional Configuration Subspaces: An Approach Based on Subgroup Discovery" submitted to the FSE 2026. 

## Running Subgroup Discovery
All datasets, subgroup discovery methods and experiment scripts needed to replicate our evaluation are provided in this repository. The required Python packages are listed in the ```requirements.txt``` file.

The ```run.ipynb``` Jupyter Notebook can be used to try various subgroup discovery methods on the data used in the paper.

## Folder Structure

- *data*
  Performance measurements collected by Mühlbauer et al. and Kaltenecker et al
- *experiments*
  An individual script for each experiment conducted in the paper
- *figures*
  Plots and tables used in the paper as well as the code to generate them
- *results*
  Results from our evaluation. In addition to the results presented in the paper, we include results for CART as well as real-world-subspaces found by each method
- *scripts*
  Helper scripts to run our evaluation on a SLURM cluster
- *src*
  The implementations for all subgroup discovery methods used in the paper as well as the code required for our evaluation

## Additional Results for RQ2

We provide results for all methods compared in the paper on real-world data. As RSD optimizes for finding complementary sets of subgroups, we allow up to 15 subgroups for that method. For all other methods, we only provide the first 5 subgroups found.

## CART

As noted in the paper, we provide additional results for CART. We implemented a subgroup discovery approach based on CART, treating each node in the regression tree as a potential subgroup and selecting those with the highest Kullback-Leibler divergence. 
We ran our experiments for RQ1 (excluding scalability) using CART, visualized in ```figures/out```, and used CART on real world data, visualized in ```figures/outs/rq2_real_world/cart_*```. 

CART achieves high F1 scores in RQ1, even outperforming Syflow and RSD in many settings. However, this result mainly stems from the fact that our experiment design was not designed with CART in mind. We seed subspaces in otherwise randomized data, leading to a mean shift in performance for exactly the configuration option involved in the seeded subspace. Furthermore, the seeded subspaces are independent and do not overlap, allowing CART to split on exactly the options that describe a certain subspace without being locked out of another. As a result, our experiments are a best-case scenario for CART and are unable to expose the weaknesses it has as a subgroup discovery method.

Qualitative analysis on our real-world results for CART clearly shows the weaknesses it has as a subgroup discovery method. CART identifies mostly redundant subspaces, all dominated by the initial split, and fails to diversify, achieving a high overlap between subspaces with low total coverage. Moreover, it tends to uncover unimodal subspaces covering only extrema of the performance distribution, missing broader patterns. So, while CART performs well in synthetic settings (RQ1), these results do not generalize, leading us to exclude CART from the our evaluation in the paper.