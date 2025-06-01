from Data import *
from address import *
from modelGen import *
from utils import LOF_context_score
from models import CondLatentCF, latentCF, latentCFpp, PrototypeLatentCF
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
import random
import tensorflow as tf

SEED = 12

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.ion()

CF_METHODS = {  "LatentCF":latentCF,
                "LatentCFpp":latentCFpp,
                "Prototype": PrototypeLatentCF,
                "CACTUS": CondLatentCF}


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
            "epochs": 200,
            "tol": 0.01,
            "target_prob": 0.5,
            "learning_rate":0.01,
            "power":0.5,
            "alpha": 0.5,
            "gamma":0.1,
            "beta": 0.01,
            "dynamicAlpha":False
        }]


# Loading data
CLASS_CONFIG_PATH = EXP[0]["classifier"]

class_config = get_exp_config(CLASS_CONFIG_PATH)
data   = getData(class_config)

#Getting classifier
classifier = modelGen(class_config["type"],data,params=class_config,debug=True)
classifier.load()

# Selecting data 
X = data.X_test
Y = data.y_test
context = data.context_test
context_labels = ["Bold","Light"]

X_0 = []; Y_0 = []; context_0 = []

N = 5
IDX = []
for val,lbl in enumerate(context_labels):
    idx =np.argwhere(context["Light"].values ==val)[:,0]
    idx = np.random.choice(idx,5)
    X_0.append(X[idx,...])
    Y_0.append(Y[idx,...])
    context_0.append(context.values[idx,...])
    IDX = IDX+idx.tolist()

X_0 = np.vstack(X_0); Y_0 = np.vstack(Y_0); context_0 = np.vstack(context_0)


# Generating the CF samples


# Getting AE-based Model
AE_CONFIG_PATH = EXP[0]["AE"]
AE_config = get_exp_config(AE_CONFIG_PATH)
aeModel = modelGen(AE_config["type"],data,params=AE_config,debug=True)
aeModel.load()


Y_0_ = np.argmax(Y_0,axis=1)

Y_0_logit = classifier.predict(X_0)


# CF generation
CF_method = CF_METHODS[EXP[0]["CFmethod"]](classifier = classifier,
                                        gen = aeModel,
                                        params = EXP[0],
                                        x = data.X_train,
                                        y = data.y_train) 


X_k,Y_k_,X,Y_ = CF_method.transform(X_0,Y_0_,target_context=context_0)

X_k_preserving = X_k


context_adv = context_0
context_adv[:,0] = 1 - context_adv[:,0] 
X_k,Y_k_,X,Y_ = CF_method.transform(X_0,Y_0_,target_context=context_adv)

X_k_changing = X_k

## Figure article
x0_ctxs = ["$C_{0} = \\text{Bold}$","$C_{0} = \\text{Light}$"]
target_ctxs = ["$x_0$","$C_{cf} = C_{0}$ (preserving)","$C_{cf} \\neq C_{0}$ (changing)"]

CFs = [X_0,X_k_preserving,X_k_changing]

best_idx = [0,8]
fontsize = 20
nc = 3
nr = 2
plt.figure("Context-changing",figsize=(16,4))
plt.clf()
for ii,x0_ctx in enumerate(x0_ctxs):
    
    for jj,target_ctx in enumerate(target_ctxs):
        x = CFs[jj][[best_idx[ii]]]
        ax = plt.subplot(nr,nc,ii*nc + jj+1)
        plt.axis('off')
        plt.imshow(x.squeeze(), cmap="gray", aspect='auto', interpolation=None)
        if ii==0:
              plt.title(target_ctx,fontsize=fontsize)
        if jj==0:
            #plt.ylabel(,fontsize=14)
            ax.text(-3.45, 24.25, x0_ctx,rotation=90,fontsize=fontsize)
plt.subplots_adjust(left=0.07, right=0.97, 
                    top=0.9, bottom=0.0024, 
                    wspace=0.1, hspace=0.1)

#plt.savefig("Figs/context-changing-tmnist.pdf")