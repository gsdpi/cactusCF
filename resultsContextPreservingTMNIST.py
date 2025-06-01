from Data import *
from address import *
from modelGen import *
from utils import LOF_context_score,cleanup_gpu
from models import CondLatentCF, latentCF, latentCFpp, PrototypeLatentCF
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

plt.ion()
SEED = 12

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


CF_METHODS = {  "LatentCF":latentCF,
                "LatentCFpp":latentCFpp,
                "Prototype": PrototypeLatentCF,
                "CACTUS": CondLatentCF}


# Configuration params for all the CF models
EXP = [    

    {
        "name": "CACTUS",
        "data": "TMNIST",
        "classifier": "./exp/TMNIST_class/config.json",
        "AE": "./exp/TMNIST_CACTUS/config.json",
        "CFmethod": "CACTUS",
        "context": [
            "Light",
            "Italic"
        ],
        "epochs": 500,
        "tol": 0.01,
        "target_prob": 0.5,
        "learning_rate": 0.01,
        "power": 0.5,
        "alpha": 0.5,
        "gamma": 0.1,
        "beta": 0.01,
        "dynamicAlpha": False
    },
    {
        "name": "Prototype D",
        "data": "TMNIST",
        "classifier": "./exp/TMNIST_class/config.json",
        "AE": "./exp/TMNIST_AE/config.json",
        "CFmethod": "Prototype",
        "context": [
            "Light",
            "Italic"
        ],
        "epochs": 500,
        "kappa": 0.0,
        "c": 1.0,
        "c_steps": 5,
        "beta": 0.05,
        "gamma": 100.0,
        "theta": 100.0,
        "clip": [
            -1000,
            1000
        ],
        "feat_range": [
            0,
            1
        ],
        "lr": 0.01
    },
    {
        "name": "Prototype C",
        "data": "TMNIST",
        "classifier": "./exp/TMNIST_class/config.json",
        "AE": "./exp/TMNIST_AE/config.json",
        "CFmethod": "Prototype",
        "context": [
            "Light",
            "Italic"
        ],
        "epochs": 500,
        "kappa": 0.0,
        "c": 1.0,
        "c_steps": 5,
        "beta": 0.05,
        "gamma": 0.0,
        "theta": 100.0,
        "clip": [
            -1000,
            1000
        ],
        "feat_range": [
            0,
            1
        ],
        "lr": 0.01
    },

    {
        "name": "LatentCF++",
        "data": "TMNIST",
        "classifier": "./exp/TMNIST_class/config.json",
        "AE": "./exp/TMNIST_AE/config.json",
        "CFmethod": "LatentCFpp",
        "context": [
            "Light",
            "Italic"
        ],
        "epochs": 500,
        "tol": 0.001,
        "target_prob": 0.5,
        "learning_rate": 0.01
    }
    
    ]


# Loading data
CLASS_CONFIG_PATH = EXP[0]["classifier"]

class_config = get_exp_config(CLASS_CONFIG_PATH)
data   = getData(class_config)

# Loading classifier

# Getting classifier
classifier = modelGen(class_config["type"],data,params=class_config,debug=True)
classifier.load()


# Selecting data 
N = 5
X = data.X_test
Y = data.y_test
context = data.context_test
context_labels = ["Bold","Light"]

X_0 = []; Y_0 = []; context_0 = []


for val,lbl in enumerate(context_labels):
    idx =np.argwhere(context["Light"].values ==val)[:,0]
    idx = np.random.choice(idx,N)
    X_0.append(X[idx,...])
    Y_0.append(Y[idx,...])
    context_0.append(context.values[idx,...])

X_0 = np.vstack(X_0); Y_0 = np.vstack(Y_0); context_0 = np.vstack(context_0)


X_k = []; Y_k_ = []; X=[]; Y_=[]

for jj, exp in enumerate(EXP):

    # Getting AE-based Model
    AE_CONFIG_PATH = exp["AE"]
    AE_config = get_exp_config(AE_CONFIG_PATH)
    aeModel = modelGen(AE_config["type"],data,params=AE_config,debug=True)
    aeModel.load()



    Y_0_ = np.argmax(Y_0,axis=1)
    Y_0_logit = classifier.predict(X_0)
    # CF generation
    CF_method = CF_METHODS[exp["CFmethod"]](classifier = classifier,
                                            gen = aeModel,
                                            params = exp,
                                            x = data.X_train,
                                            y = data.y_train) 
    # CF generation
    x_k,y_k_,x,y_ = CF_method.transform(X_0,Y_0_,target_context=context_0)

    X_k.append(x_k); Y_k_.append(y_k_); X.append(x); Y_.append(y_)
    cleanup_gpu()

## Article image 

plt.figure("context-preserving-TMNIST",figsize=(10,4)); plt.clf()

nc = len(EXP)+1
nr = 2

IDX = [4,5]
context_labels = ["$C_{cf} = \\text{Bold}$","$C_{cf} = \\text{Light}$"]
fontsize=16

for jj, exp in enumerate(EXP):
    for i,idx in enumerate(IDX):
        x_0 = X_0[[idx]]
        y_0 = Y_0_[idx]
        cntxt_0 = context_0[[idx]]
        
        x_k = X_k[jj][[idx]]
        y_k_logit = classifier.predict(x_k)
        y_k_logit = float(y_k_logit[0,1-y_0])
        
        if jj ==0:
            ax = plt.subplot(nr,nc,i*nc + jj+1)
            if i ==0:
                plt.title(f"$x_0$",fontsize=14)
            plt.imshow(x_0.squeeze(), cmap="gray", aspect='auto', interpolation=None)
            ax.text(-5.45, 24.25, context_labels[i],rotation=90,fontsize=fontsize)
            plt.axis('off')


        ax = plt.subplot(nr,nc,i*nc + jj+2)
        if i ==0:
            plt.title(f"{exp['name']}",fontsize=14)
        plt.imshow(x_k.squeeze(), cmap="gray", aspect='auto', interpolation=None)
        plt.axis('off')

plt.tight_layout()
#plt.savefig("Figs/context-preserving-tmnist.pdf")