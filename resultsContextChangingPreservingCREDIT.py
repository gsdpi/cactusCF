# SCRIPT DE PRUEBA PARA LA GENERACIÓN DE CF CON CATUS EN CREDIT

from Data import *
from address import *
from modelGen import *
from utils import LOF_context_score,cleanup_gpu
from models import CondLatentCF, latentCF, latentCFpp, PrototypeLatentCF
import numpy as np
import matplotlib.pyplot as plt

import random
import tensorflow as tf

SEED = 12

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



plt.ion()

# Métodos de CF
CF_METHODS = {  "LatentCF":latentCF,
                "LatentCFpp":latentCFpp,
                "Prototype": PrototypeLatentCF,
                "CACTUS": CondLatentCF}

# Experimentos que a realizar
EXP = [    

    {
        "name": "CACTUS",
        "data": "GIVECREDIT",
        "classifier": "./exp/GIVECREDIT_class/config.json",
        "AE": "./exp/GIVECREDIT_CACTUS/config.json",
        "CFmethod": "CACTUS",
        "context": [
            "ageAbove50",
            "Dependents"
        ],
        "epochs": 300,
        "target_prob": 0.5,
        "learning_rate": 0.01,
        "power": 0.5,
        "alpha": 0.7,
        "gamma": 0.1,
        "beta": 0.01,
        "dynamicAlpa": False
    },
    
        {
        "name": "Prototype D",
        "data": "GIVECREDIT",
        "classifier": "./exp/GIVECREDIT_class/config.json",
        "AE": "./exp/GIVECREDIT_AE/config.json",
        "CFmethod": "Prototype",
        "context": [
            "ageAbove50",
            "Dependents"
        ],
        "epochs": 100,
        "kappa": 0.0,
        "c": 1.0,
        "c_steps": 1,
        "beta": 0.1,
        "gamma": 100.0,
        "theta": 100.0,
        "clip": [
            -1000,
            1000
        ],
        "feat_range": [
            -4,
            4
        ],
        "lr": 0.01
    },

        {
        "name": "Prototype C",
        "data": "GIVECREDIT",
        "classifier": "./exp/GIVECREDIT_class/config.json",
        "AE": "./exp/GIVECREDIT_AE/config.json",
        "CFmethod": "Prototype",
        "context": [
            "ageAbove50",
            "Dependents"
        ],
        "epochs": 100,
        "kappa": 0.0,
        "c": 1.0,
        "c_steps": 1,
        "beta": 0.1,
        "gamma": 0.0,
        "theta": 100.0,
        "clip": [
            -1000,
            1000
        ],
        "feat_range": [
            -4,
            4
        ],
        "lr": 0.01
    },
       {
        "name": "LatentCF++",
        "data": "GIVECREDIT",
        "classifier": "./exp/GIVECREDIT_class/config.json",
        "AE": "./exp/GIVECREDIT_AE/config.json",
        "CFmethod": "LatentCFpp",
        "context": [
            "ageAbove50",
            "Dependents"
        ],
        "epochs": 100,
        "tol": 0.001,
        "target_prob": 0.5,
        "learning_rate": 0.1
    }
    ]



# Loading data
CLASS_CONFIG_PATH = EXP[0]["classifier"]
class_config = get_exp_config(CLASS_CONFIG_PATH)
data   = getData(class_config)

# Getting the classifier
classifier = modelGen(class_config["type"],data,params=class_config,debug=True)
classifier.load()


# Selecting data 
X = data.X_test
Y = data.y_test
context = data.context_test

context_labels = ["Older 50","Younger 50"]


X_0 = []; Y_0 = []; context_0 = []; LOF_0=[]
N = 5
IDX = []
for val,lbl in enumerate(context_labels):
    idx =np.argwhere(context["ageAbove50"].values ==val)[:,0]
    idx = np.random.choice(idx,5)
    IDX = IDX + idx.tolist()
    X_0.append(X[idx,...])
    Y_0.append(Y[idx,...])
    context_0.append(context.values[idx,...])
    

X_0 = np.vstack(X_0); Y_0 = np.vstack(Y_0); context_0 = np.vstack(context_0)
LOF_0 = LOF_context_score(data.X_train,data.context_train.values,X_0,context_0,agg=False)


#########
# Context-preserving CF generation
#########

X_k = []; Y_k_ = []; X=[]; Y_=[]; LOF_k = []

for jj, exp in enumerate(EXP):

    # Generating the CF samples
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
    
    lof_k = LOF_context_score(data.X_train,data.context_train.values,x_k,context_0,agg=False)   
         
    
    x_k = data.scaler_inverse_transform(x_k)
    
    X_k.append(x_k); Y_k_.append(y_k_); X.append(x); Y_.append(y_);LOF_k.append(lof_k)
    cleanup_gpu()

#########
# Context-changing CF generation
#########
context_ch = context_0
context_ch[:,0] = 1 - context_ch[:,0] 

X_k_ch = []; Y_k_ch_ = []; X_ch=[]; Y_ch_=[]; LOF_k_ch = []

for jj, exp in enumerate(EXP):

    # Generating the CF samples
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
    x_k_ch,y_k_ch_,x_ch,y_ch_ = CF_method.transform(X_0,Y_0_,target_context=context_ch)
    
    lof_k_ch = LOF_context_score(data.X_train,data.context_train.values,x_k_ch,context_ch,agg=False)   
         
    
    x_k_ch = data.scaler_inverse_transform(x_k_ch)
    
    X_k_ch.append(x_k_ch); Y_k_ch_.append(y_k_ch_); X_ch.append(x_ch); Y_ch_.append(y_ch_);LOF_k_ch.append(lof_k_ch)
    cleanup_gpu()








X_0 = data.scaler_inverse_transform(X_0)
Y_0 =  np.argmax(Y_0,axis = 1)





# #####################################################################################

# # CONTEXT-PRESERVING

# #####################################################################################

# X_k,Y_k_,X,Y_ = CF_method.transform(X_0,Y_0_,target_context=context_0)


# # Showing an original sample
# x0 = data.scaler_inverse_transform(X_0[[0]])[0]
# xk = data.scaler_inverse_transform(X_k[[0]])[0]
# y0 =  np.argmax(Y_0[0])
# yk =  np.argmax(Y_k_[0])


from utils import styling_mpl_table



plt.figure("Under 50",figsize=(16, 9)); plt.clf()

nr=5
colLabels = [None,"$x_0$"] + [exp["name"] for exp in EXP]
table_data = []
for ii,sample in enumerate(np.arange(0,5,1)):
    
    x0 = X_0[sample]
    y0 = Y_0[sample]
    lof_0 = LOF_0[sample]
    
    ax =plt.subplot(3,2,ii+1)
    plt.axis('off')   
    #["Feature","Value"] + 
    table_data = [[data.features_lbls[feat], round(x0[feat],2)]+[round(X_k[n_exp][sample][feat],2) for n_exp in range(len(EXP))] for feat in range(len(x0)) ]    #[data.features_lbls] + [x0[0].tolist()] 
    table_data = table_data + [[None]*len(colLabels)]
    table_data = table_data + [["Predicted class",y0] + [np.argmax(Y_k_[n_exp][sample]) for n_exp in range(len(EXP))] ]
    table_data = table_data + [["$\\text{LOF}_{10}$",round(lof_0,2)] + [round(LOF_k[n_exp][sample],2) for n_exp in range(len(EXP))] ]

    table = plt.table(cellText=table_data, loc='center', colLabels=colLabels, colLoc='left', cellLoc='left')


    styling_mpl_table(table,table_data)

    plt.title(f"Idx: {IDX[sample]}",fontdict={"fontsize":9},loc="left")
    #plt.tight_layout()
plt.show()
plt.tight_layout()




plt.figure("Above 50",figsize=(16, 9)); plt.clf()

nr=5
colLabels = [None,"$x_0$"] + [exp["name"] for exp in EXP]
table_data = []
for ii,sample in enumerate(np.arange(5,10,1)):
    
    x0 = X_0[sample]
    y0 = Y_0[sample]
    cntxt_0 = context_0[[sample]]
    lof_0 = LOF_0[sample]
    

    ax =plt.subplot(3,2,ii+1)
    plt.axis('off')   
    #["Feature","Value"] + 
    table_data = [[data.features_lbls[feat], round(x0[feat],2)]+[round(X_k[n_exp][sample][feat],2) for n_exp in range(len(EXP))] for feat in range(len(x0)) ]    #[data.features_lbls] + [x0[0].tolist()] 
    table_data = table_data + [[None]*len(colLabels)]
    table_data = table_data + [["Predicted class",y0] + [Y_k_[n_exp][sample] for n_exp in range(len(EXP))] ]
    table_data = table_data + [["$\\text{LOF}_{10}$",round(lof_0,2)] + [round(LOF_k[n_exp][sample],2) for n_exp in range(len(EXP))] ]
    
    
    #table_data = table_data + [["Prediction",y0]]

    table = plt.table(cellText=table_data, loc='center', colLabels=colLabels, colLoc='left', cellLoc='left')

    styling_mpl_table(table,table_data)

    plt.title(f"Idx: {IDX[sample]}",fontdict={"fontsize":9},loc="left")
    #plt.tight_layout()
plt.show()
plt.tight_layout()


###########################################################################
# Figures article
###########################################################################


colLabels = ["$\\mathbf{x_0}$"] + [exp["name"] for exp in EXP]
table_data = []
best_idx = [2016]
best_idx = [IDX.index(i) for i in best_idx]

id_context = ["$C_0$ = $C_{cf}= \\text{Under 50}$","Above 50"]

for ii,sample in enumerate(best_idx):
    
    x0 = X_0[sample]
    y0 = Y_0[sample]
    cntxt_0 = context_0[[sample]]
    lof_0 = LOF_0[sample]
    
    
    table_data = [[data.features_lbls[feat], round(x0[feat],2)]+[round(X_k[n_exp][sample][feat],2) for n_exp in range(len(EXP))] for feat in range(len(x0)) ]    #[data.features_lbls] + [x0[0].tolist()] 
    table_data = table_data + [["Predicted class",y0] + [Y_k_[n_exp][sample] for n_exp in range(len(EXP))] ]
    columns  = [id_context[ii]]+colLabels
    df_table = pd.DataFrame(table_data,columns=columns)
    df_table.index = df_table[columns[0]]; df_table.drop(labels=columns[0],axis=1,inplace=True)
    
latex_code = df_table.to_latex(
    index=True,
    column_format='llllll',
    escape=False,
    float_format="%.1f",
    caption="Context-preserving ",
    label="tab:comparacion_modelos"
)



# Saving the latex code
with open("./Figs/context-preserving-credit.tex", "w", encoding="utf-8") as f:
    f.write(latex_code)





#Context-changing

x0_ctxs = ["Under 50","Above 50"]
#target_ctxs = ["$\\mathbf{x_0}$","$\\mathbf{C_\\text{target} = \\text{Above 50}}$","$\\mathbf{C_\\text{target} = \\text{Under 50}}$"]
target_ctxs = ["$\\mathbf{x_0}$","Preserving","Changing"]
colLabels = ["$\\mathbf{x_0}$","Preserving","Changing"] + [exp["name"] for ii,exp in enumerate(EXP) if ii !=0]


CFs = [X_0,X_k[0],X_k_ch[0]] + [X_k[i] for i in range(1,len(X_k),1)]
CFs_y_ = [Y_0,Y_k_[0],Y_k_ch_[0]] + [Y_k_[i] for i in range(1,len(X_k),1)]
best_idx = 4

sample = best_idx
    
table_data = [[data.features_lbls[feat]]+[round(CFs[n][sample][feat],2) for n in range(len(CFs))] for feat in range(len(x0)) ]    #[data.features_lbls] + [x0[0].tolist()] 
table_data = table_data + [["Predicted class"] + [CFs_y_[n][sample] for n in range(len(CFs))] ]

columns  = ["$\\mathbf{C_0 = \\text{Under 50}}$"]+colLabels
df_table = pd.DataFrame(table_data,columns=columns)
df_table.index = df_table[columns[0]]; df_table.drop(labels=columns[0],axis=1,inplace=True)


latex_code = df_table.to_latex(
    index=True,
    column_format='llllll',
    escape=False,
    float_format="%.1f",
    caption="Context-changing ",
    label="tab:comparacion_modelos"
)



# Saving the latex code
# with open("./Figs/context-changing-preserving-credit.tex", "w", encoding="utf-8") as f:
#     f.write(latex_code)





#Context-changing