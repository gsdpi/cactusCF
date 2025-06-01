
import ipdb
from Data import *
from address import *
from modelGen import *
from utils import cf_eval,cleanup_gpu
from models import CondLatentCF, latentCFpp, PrototypeLatentCF
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
                METRICS.append([exp["name"],exp["data"],exp["CFmethod"],label, cf_scores[i],exp["alpha"],"preserving"])
            send_telegram_message(f"Trial: {trial}/{N_BOOTSTRAP})")
        # Cleaning GPU models
        cleanup_gpu()

        # Context adverse scenario
        for trial in range(N_BOOTSTRAP):
            # Getting the CFs
            rand_idx = np.random.choice(len(data.X_test),N,replace=True)
            x = data.X_test[rand_idx,...]
            context = data.context_test[exp['context']].values; context = context[rand_idx,...] 
            context_training = data.context_train[exp['context']].values
            y = np.argmax(data.y_test[rand_idx,...],axis=1)
            y_ = np.argmax(classifier.predict(x),axis=1)

            context_adv = context.copy()
            context_adv[:,1] = 1 - context_adv[:,1] 
            cf_x,cf_y_,x,__ = CF_method.transform(x,y_,target_context=context_adv)

            cf_scores,cf_scores_labels = cf_eval(data.X_train, context_training,x,cf_x, y_,context_adv,cf_y_)

            for i,label in enumerate(cf_scores_labels):
                METRICS.append([exp["name"],exp["data"],exp["CFmethod"],label, cf_scores[i],exp["alpha"],"changing"])
            
        # Cleaning GPU models
        cleanup_gpu()

    df_metrics = pd.DataFrame(data = METRICS, columns = ["Model","Dataset", "CFMethod", "Metric", "Value","Alpha","preserving/changing"])
    df_metrics = df_metrics.round(3)
    print(df_metrics)


    if store:
        # Save CSV and LaTeX files
        with open(os.path.join(PATH,"CFAblation.csv"), "w") as f:
            f.write(df_metrics.to_csv(index=False))



if __name__ == "__main__":

    try:

        args = parse_args()
        config = get_exp_config(args.config)
        PATH = os.path.dirname(args.config)
        N = args.N
        N_BOOTSTRAP = args.N_BOOTSTRAP if args.N_BOOTSTRAP >= 1 else  2
        print(config)
        evaluate_models(config, N,N_BOOTSTRAP,PATH,store = args.store)

    except Exception as e:
        print(f"Error: {e}")  # Optional: Print to console
        raise  # Re-raises the original exception
