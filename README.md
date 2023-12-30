# Skateboarding Trick Classifier

[![DOI](https://zenodo.org/badge/422018559.svg)](https://zenodo.org/doi/10.5281/zenodo.6989815)

<img src="./data/skateboarding-logo.jfif" alt="" height="400">

This repository contains _accelerometry signals_ from a skateboard mounted with an accelerometer/recorder. The accelerometer was used to record several skateboarding maneuvers from 5 different classes. To solve the classification task we trained a neural network with our dataset. We trained both a flat-dense and a recurrent network (LSTM). Ensemble models for the 'flat-dense' and 'rnn' architectures were also trained. The dataset can be found in the `data` folder and models in the `skateboarding_models` folder. You can also follow the procedure with our `Skateboarding_Trick_Classifier` notebook.

Article: _[Development of a skateboarding trick classifier using accelerometry and machine learning](https://www.scielo.br/j/reng/a/sgsxHt4HffBYxDhqj9QD3dS/abstract/?lang=en)_.

## Requirements

---

```bash

ipython
keras
numpy
pandas
plotly
scikit_learn
scipy
tensorflow

```

## Cite as 🤗

---

```latex

@article{correa2017development,
  title={Development of a skateboarding trick classifier using accelerometry and machine learning},
  author={Corr{\^e}a, Nicholas Kluge and Lima, J{\'u}lio C{\'e}sar Marques de and Russomano, Thais and Santos, Marlise Araujo dos},
  journal={Research on Biomedical Engineering},
  volume={33},
  pages={362--369},
  year={2017},
  publisher={SciELO Brasil}
}

```
