# 


# CACTUS: Context-Aware Counterfactual Explanations

This repository contains the official implementation of **CACTUS**, the framework introduced in the article:

> **CACTUS: A Context-aware Framework for Counterfactual Explanations Across Diverse Prediction Domains**  
> [Diego García, Zhendong Wang, José M. Enguita]  



---

## 🧠 Abstract

Counterfactual explanations offer actionable insights for black-box classifiers by suggesting minimal changes that yield desirable prediction outcomes. However, most methods overlook contextual integrity—such as demographic consistency or user-defined constraints.
CACTUS introduces a novel framework for generating feasible counterfactuals that either preserve or modify user-defined contextual features. It operates in a context-conditional latent space using a composite β-VAE model to disentangle context-related factors. Our method demonstrates superior contextual consistency while maintaining competitive performance on key counterfactual metrics.

---
## 🧾 Data
---

Two datasets were employed in the empirical evaluation of CACTUS:

1.  TMNIST (A database of Typeface based digits):

    https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist 
2.  Give Me Some Credit:

    https://www.kaggle.com/c/GiveMeSomeCredit

To reproduce the experiments, the source data files in the links shoul be downloaded and processed by the scripts in [Data/](./Data/)
## 🗂️ Repository Structure

```plaintext
├── Data/                                     # Folder containing datasets and preprocessing scripts
│   ├── DataReader_GivmeCred.py/            
│   ├── DataReader_TMNIST.py/ 
├── models/                                     # Model architecture definitions (e.g., β-VAE, classifiers)
│   ├── AE.py                                   # Standard Autoencoder for latent space learning
│   ├── BaseModel.py                            # Base class for shared model functionality
│   ├── CACTUS_VAE_2D.py                        # CACTUS model for 2D image datasets (e.g., TMNIST)
│   ├── CACTUS_VAE_tabular.py                   # CACTUS model for tabular datasets (e.g., CREDIT)
│   ├── CNN_2D.py                               # 2D CNN used for classification on images
│   ├── CNNAE_2D.py                             # Convolutional Autoencoder for 2D data
│   ├── CNN.py                                  # Generic CNN architecture
│   ├── CondLatentCF.py                         # CACTUS model for CF generation
│   ├── DNN.py                                  # Dense Neural Network for tabular data
│   ├── latentCFpp.py                           # LatentCF++ model for CF generation
│   ├── PrototypeLatentCF.py                    # ProtoCF model
├── exp/                                        # Experiment configurations and results
├── address.py                                  # Utilities related to addressing and path routing 
├── utils.py                                    # Utility functions for logging, metrics, etc.
├── modelGen.py                                 # Model generation and loading utilities
├── train.py                                    # Main training entry point
├── CFResults.py                                # Analysis and metrics for counterfactual 
├── AblationStudy.py                            # Script for running ablation study experiments
├── resultsAblationStudyFigure.py               # Script to generate ablation figures
├── resultsContextChangingPreservingCREDIT.py   # Visualizations for CREDIT dataset (context changing/preserving)
├── resultsContextChangingTMNIST.py             # Visualizations for TMNIST (context changing)
├── resultsContextPreservingTMNIST.py           # Visualizations for TMNIST (context preserving)
├── trainAEs.sh                                 # Shell Script to train autoencoders
├── trainClassifiers.sh                         # Shell Script to train classifiers
├── cfEvaluation.sh                             # Shell Script to evaluate counterfactual results
├── ablationStudy.sh                            # Shell script to automate ablation runs
├── requirements.yml                            # Conda environment dependencies
├── README.md                                   # Project documentation
```

## ✉️ Contact
For questions or collaborations, feel free to contact:

[Diego García]() – [garciaperdiego@uniovi.es](garciaperdiego@uniovi.es)