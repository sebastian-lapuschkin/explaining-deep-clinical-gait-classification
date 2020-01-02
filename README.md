# On the Understanding and Interpretation of Machine Learning Predictions in Clinical Gait Analysis Using Explainable Artificial Intelligence *

![overview figure](./figures/overview/overview_46.png)

This repository contains the python code for training and evaluation of models as presented in
[On the Understanding and Interpretation of Machine Learning Predictions in Clinical Gait Analysis Using Explainable Artificial Intelligence](https://arxiv.org/abs/1912.07737)
```
@article{horst2019understanding,
  title={On the Understanding and Interpretation of Machine Learning Predictions in Clinical Gait Analysis Using Explainable Artificial Intelligence},
  author={Horst, Fabian and
          Slijepcevic, Djordje and
          Lapuschkin, Sebastian and
          Raberger, Anna-Maria and
          Zeppelzauer, Matthias and
          Samek, Wojciech and
          Breiteneder, Christian and
          Sch{\"o}llhorn, Wolfgang I
          and Horsak, Brian},
  journal={arXiv preprint arXiv:1912.07737},
  year={2019}
}
```


Folder `figures` contains code and data for (generating) the overview figure shown in the paper.

Folder `python` contains code for model training and evaluation, based on python3 and the python sub-package of the [LRP Toolbox (version 1.3.0rc2)](https://github.com/sebastian-lapuschkin). Should you use or extend this implementation please consider citing the toolbox, as well as our paper mentioned above.
```
@article{lapuschkin2016toolbox,
    author  = {Lapuschkin, Sebastian and Binder, Alexander and Montavon, Gr{\'e}goire and M\"uller, Klaus-Robert and Samek, Wojciech},
    title   = {The LRP Toolbox for Artificial Neural Networks},
    journal = {Journal of Machine Learning Research},
    year    = {2016},
    volume  = {17},
    number  = {114},
    pages   = {1-5},
    url     = {http://jmlr.org/papers/v17/15-618.html}
}
```

In folder `python`, the file `install.sh` contains instructions to setup [`Miniconda3`](https://docs.conda.io/en/latest/miniconda.html)-based virtual environments for python, as required by our code.
Option A only considers CPU hardware, while option B enables GPU support
for neural network training and evaluation. Comment/uncomment the lines appropriately.

All recorded gait data used in the paper is available in folder `python/data`.
Training- and evalation scripts for fully reproducing the data splits, models and prediction explanations are
provided with with files `python/gait_experiments_batch*.py`.
The folder `sge` contains files `*.args`, presenting the mentioned training-evaluation runs as (probably more) handy command line parameters, one per line, either to be called directly as
```
python gait_experiments ${ARGS_LINE}
```
or to be submitted to a SUN Grid Engine with
```
python sge_job_simple.py your_file_of_choice.args
```
Some paths and variables need to be adjusted.


