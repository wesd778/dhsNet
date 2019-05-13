# dhsNet

#### Deep learning for DNase I hypersensitive sites (DHSs) identification.

---------------------------------------------------------------------------------------------------
### Requirement

Python 2.7.14; Pytorch 0.4.1; scikit-learn 0.19.1

---------------------------------------------------------------------------------------------------
### Tutorials

Train a new model, run `./train.py`. The hyper-parameters can be found in `./config.py`. Please change the species in `./train.py` before training.

Well trained multi-scale models are stored in `./output/model/`. 

Single-scale training is also allowed, `./TAIR10_DHSs.fas` and `./TAIR10_Non_DHSs.fas` are arabidopsis datasets downloaded from [pdhs-elm](https://link.springer.com/article/10.1007%2Fs00438-018-1436-3).
