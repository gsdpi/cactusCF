
import ipdb
from Data import *
from address import *
from modelGen import *
from utils import cf_eval,cleanup_gpu
from models import CondLatentCF,  latentCFpp, PrototypeLatentCF
import numpy as np
import random 
from tensorflow import random as tf_random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
import argparse
plt.ion()


SEED = 23

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf_random.set_seed(SEED)




CF_METHODS = {  
                "LatentCFpp":latentCFpp,
                "Prototype": PrototypeLatentCF,
                "CACTUS": CondLatentCF}



def parse_args():
    parser = argparse.ArgumentParser(description="CF generation script")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--store", action='store_true', help="store the metrics in config PATH")
    parser.add_argument("--N", type=int, default=100,help="number of test samples")
    parser.add_argument("--N_BOOTSTRAP", type=int, default=10,help="number of trials")
    return parser.parse_args()


def evaluate_models(EXP,N,N_BOOTSTRAP,PATH,store=True):
    # Number of samples to compute the metrics for CF evaluation
    METRICS = []

    for i,exp in enumerate(EXP):
        print("\n"*2)
        print("*"*200)
        print(f"Running exp: {exp['name']} ({i}/{len(EXP)})")
        print("*"*200)
        print("\n"*2)
        
        # Reading the data
        CLASS_CONFIG_PATH = exp["classifier"]
        AE_CONFIG_PATH = exp["AE"]

        class_config = get_exp_config(CLASS_CONFIG_PATH)
        print("Reading data")
        data   = getData(class_config)

        # Getting classifier
        classifier = modelGen(class_config["type"],data,params=class_config,debug=True)
        classifier.load()
        # Getting AE-based Model
        AE_config = get_exp_config(AE_CONFIG_PATH)
        aeModel = modelGen(AE_config["type"],data,params=AE_config,debug=True)
        aeModel.load()
        # CF generation
        CF_method = CF_METHODS[exp["CFmethod"]](classifier = classifier,
                                                gen = aeModel,
                                                params = exp,
                                                x = data.X_train,
                                                y = data.y_train)    

        for trial in range(N_BOOTSTRAP):
            # Getting the CFs
            rand_idx = np.random.choice(len(data.X_test),N,replace=True)
            x = data.X_test[rand_idx,...]
            context = data.context_test[exp['context']].values; context = context[rand_idx,...] 
            context_training = data.context_train[exp['context']].values
            y = np.argmax(data.y_test[rand_idx,...],axis=1)
            y_ = np.argmax(classifier.predict(x),axis=1)
            cf_x,cf_y_,x,__ = CF_method.transform(x,y_,target_context=context)
            cf_scores,cf_scores_labels = cf_eval(data.X_train, context_training,x,cf_x, y_,context,cf_y_)

            for i,label in enumerate(cf_scores_labels):
                METRICS.append([exp["name"],exp["data"],exp["CFmethod"],label, cf_scores[i]])
            send_telegram_message(f"Trial: {trial}/{N_BOOTSTRAP})")
        # Cleaning GPU models
        cleanup_gpu()

    df_metrics = pd.DataFrame(data = METRICS, columns = ["Model","Dataset", "CFMethod", "Metric", "Value"])
    df_metrics = df_metrics.round(3)
    print(df_metrics)



     # Pivoting the dataframe
    df_metrics_pivot = df_metrics.pivot_table(index=["Dataset", "Model"], columns="Metric", values="Value",aggfunc=['mean','std'])    

    #df_metrics_pivot = pd.read_csv("exp/CF_EVALUATION/CFevaluation_best.csv",index_col=[0,1],header=[0,1])

    desired_order = ['validity', 'lof_context', 'compactness', 'n_proximity']
    df_metrics_pivot.columns = df_metrics_pivot.columns.swaplevel(0, 1)
    df_metrics_pivot = df_metrics_pivot[desired_order]
    df_metrics_pivot.rename(columns={"'validity'": "Validity",
                                      "lof_context": "LOF",
                                      "compactness":"Compactness",
                                      "n_proximity":"Proximity"},
                                       inplace=True)
    df_metrics_pivot.columns = df_metrics_pivot.columns.swaplevel(0, 1)

    df_formatted = df_metrics_pivot.apply(
    lambda x: x["mean"].map("${:2.2f}".format) + " \pm " + x["std"].map("{:2.2f} $".format), axis=1
    )


    # Converting the dataframe to LaTeX with multirow formatting
    latex_output = df_formatted.to_latex(index=True, multirow=True,escape=False)
    print(latex_output)
    # Resetting index to ensure proper format
    df_metrics_pivot = df_metrics_pivot.reset_index()

    print(df_metrics_pivot)
    # Converting the dataframe to CSV
    csv_output = df_metrics_pivot.to_csv(index=False)


    # Display the dataframe for verification
    
    if store:
        # Save CSV and LaTeX files
        with open(os.path.join(PATH,"CFevaluation.csv"), "w") as f:
            f.write(csv_output)

        with open(os.path.join(PATH,"CFevaluation.tex"), "w") as f:
            f.write(latex_output)

    # Restoring the metrics
    #df_metrics_pivot = pd.read_csv("CFevaluation.csv",index_col=[0,1],header=[0,1])


if __name__ == "__main__":

    try:
        

        args = parse_args()
        config = get_exp_config(args.config)
        PATH = os.path.dirname(args.config)
        N = args.N
        N_BOOTSTRAP = args.N_BOOTSTRAP if args.N_BOOTSTRAP > 1 else  2

        print(config)


        evaluate_models(config, N,N_BOOTSTRAP,PATH,store = args.store)
        
    except Exception as e:
        
        print(f"Error: {e}")  # Optional: Print to console
        raise  # Re-raises the original exception
