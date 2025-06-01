from .DataReader import DataReader
from .DataReader_TMNIST import DataReader_TMNIST
from .DataReader_GivmeCred import DataReader_GivmeCred
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from address import *
from utils import upsampling

def getData(params):
    
    if params["data"]=="ECG":
        PATH = get_data_path("PTB-XL")
        data = DataReader(PATH,100,leads = params["leads"],clss = params["clss"],timesteps=params["timesteps"],from_file=PATH+"PTB-XL.h5")

        idx_y_lbls = [data.cls_lbls_all.index(lbl) for lbl in params["clss"]]
        data.y_train = data.y_train[:,idx_y_lbls]
        data.y_test = data.y_test[:,idx_y_lbls]

        data.X_train, data.y_train,up_idx = upsampling(data.X_train, data.y_train)

        data.context_train = data.context_train.iloc[up_idx,:]

        return data
    
    elif params["data"]=="TMNIST":
        PATH = get_data_path("TMNIST")
        data = DataReader_TMNIST(PATH,binary_class=params['class'])
    
        return data
    

    elif params["data"] == "GIVECREDIT":
        PATH = get_data_path("GiveMeCredit")
        data = DataReader_GivmeCred(PATH)

        data.X_train, data.y_train, up_idx = upsampling(data.X_train, data.y_train)

        data.context_train = data.context_train.iloc[up_idx,:]
        
        return data
    else:
        raise Exception(f"Dataset is not supported yet")