# **InfEst**

**InfEst** is a Python library for estimating information measures from data.

From finite discrete data, InfEst can estimate: 

* entropy 
* joint entropy
* mutual information 
* partial information decomposition (specific, unique, redundant, and synergistic information)

For the first three in the above list (and a simplified version of the partial information decomposition), InfEst uses the [NSB method](https://arxiv.org/abs/physics/0108025) to provide less biased estimates for finite data.  In these cases, uncertainties in the estimates are also provided.


## Installation

To install the package and its python dependencies using pip, clone the repository, descend into the `InfEst/` folder, and run

```
pip install -e .
```


## Examples

For examples of code usage, see the included jupyter notebook file `examples.ipynb`.


## Contributors

Bryan C Daniels
