# Explaining the unique nature of individual gait patterns with deep learning

This repository contains the python code for training and evaluation of models as presented in
[Explaining the unique nature of individual gait patterns with deep learning](https://doi.org/10.1038/s41598-019-38748-8)

```
TODO: add paper bibtex
```


Folder `figures` contains code and data for (generating) the figures shown in the paper.
```
TODO: add content
TODO: link fancy overview figure
```

Folder `python` contains code for model training and evaluation, as a derivation of the python sub-package of the [LRP Toolbox (version 1.2.0)](https://github.com/sebastian-lapuschkin). Should you use or extend this implementation please consider citing the toolbox next to our paper mentioned above.

```
@article{JMLR:v17:15-618,
    author  = {Sebastian Lapuschkin and Alexander Binder and Gr{{\'e}}goire Montavon and Klaus-Robert M{{{\"u}}}ller and Wojciech Samek},
    title   = {The LRP Toolbox for Artificial Neural Networks},
    journal = {Journal of Machine Learning Research},
    year    = {2016},
    volume  = {17},
    number  = {114},
    pages   = {1-5},
    url     = {http://jmlr.org/papers/v17/15-618.html}
}
```

Files describing the training, validation and test splits, the trained models on different feature sets (in part not discussed in the paper) and target labels (in part not discussed in the paper), as well as the model outputs, scores and analyses obtained via LRP and perturbation analysis can be obtained from the following locations, grouped by model type:

```
TODO: Link Linear baseline cases
TODO: Link 2-layer MLPs
TODO: Link 3-layer MLPs
TODO: Link CNNs
```
