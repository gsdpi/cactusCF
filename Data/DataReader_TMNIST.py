"""
Script for reading and pre-processing the data.

#


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipdb
import h5py

class DataReader_TMNIST(object):
    def __init__(self, path,binary_class = None):
        self.name = "TMNIST_Data"
        self.path = path
        self.binary_class = binary_class
        
        self.data = pd.read_csv(path+"TMNIST_Data.csv")
        self.light_labels = ["Light"]
        self.italic_labels = ["Italic"]
        self._process_data()
        self.clss_lbls = np.unique(self.Y)
        
        self.diag = np.eye(len(np.unique(self.Y)))
        self.Y = self.diag[self.Y,:]
        
        # Split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test,self.context_train, self.context_test = train_test_split(self.X, self.Y,self.context, test_size=0.2,random_state=42)


        # ####### REPORT #######
        print("\n"*3)
        print("-"*100)
        print("---- DATA SUMMARY ----")
        for cls in self.clss_lbls:
            print(f"Class {cls}: {(self.Y==cls).sum()} samples")
        
        print(f"Light att (No true labels): {self.context.Light.sum()}")
        print(f"Italic att (No true labels): {self.context.Italic.sum()}")
        print("-"*100)
        print("\n"*3)

        return None
    
    
    def _process_data(self):
        
        # filtering samples that contains "Bold" "Regular" "Medium" "Italic"
        filt_idx = np.zeros_like(self.data["names"].values)
        for s in ["Bold", "Light"]:
            filt_idx = np.logical_or(filt_idx,self.data["names"].str.contains(s).values)
        self.data = self.data.iloc[filt_idx,:]
    
        # Getting input images
        self.X = self.data.values[:,2:].reshape((-1,28,28,1))/255.
        self.X = self.X.astype("float32")
    
        self.Y = self.data.values[:,[1]].astype("int32")

        
        # Getting context
        self.italic = self.data["names"].str.contains("Italic").values
        self.italic = self.italic.astype(int)
        self.light   = self.data["names"].str.contains("Light").values

        self.light = self.light.astype(int)
            
        self.context = pd.DataFrame(data = np.vstack([self.italic,self.light]).T,columns=["Italic","Light"])

        # Binary case. Only the user-indicated classes are used
        if self.binary_class != None:
            print("Filtering labels for binary case")
            idx = np.logical_or(self.Y==self.binary_class[0],self.Y==self.binary_class[1]).squeeze()
            
            self.Y = self.Y[idx]
            self.Y[self.Y==self.binary_class[0]] =0
            self.Y[self.Y==self.binary_class[1]] =1
            self.Y = self.Y.squeeze()
            self.X = self.X[idx]
            self.context = self.context[idx]
            
            self.clss_lbls = np.array(self.binary_class)





if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    plt.ion()
    PATH = "/home/diego/repositorios/24_03_05_condLS/Data/TMNIST/"
    print(f"Getting data from {PATH}")
    data = DataReader_TMNIST(PATH,binary_class=[1,9])
    
    #data = DataReader(PATH,100,from_file=None); data.store("./PTB-XL/PTB-XL.h5")
    N = 25
    idx_bold = np.random.choice(np.argwhere(data.context["Light"].values==1)[:,0],N)
    plt.close("Light")
    plt.figure("Light")
    for i,idx in enumerate(idx_bold):
        plt.subplot(5,5,i+1)
        plt.imshow(data.X[idx,...].squeeze(),cmap='gray',aspect='auto')
    plt.suptitle("Light")


    idx_light = np.random.choice(np.argwhere(data.context["Light"].values==0)[:,0],N)
    plt.close("bold")
    plt.figure("bold")
    for i,idx in enumerate(idx_light):
        plt.subplot(5,5,i+1)
        plt.imshow(data.X[idx,...].squeeze(),cmap='gray',aspect='auto')
    plt.suptitle("Bold")
    


    idx_italic = np.random.choice(np.argwhere(data.context["Italic"].values==1)[:,0],N)
    plt.close("italic")
    plt.figure("italic")
    for i,idx in enumerate(idx_italic):
        plt.subplot(5,5,i+1)
        plt.imshow(data.X[idx,...].squeeze(),cmap='gray',aspect='auto')
    plt.suptitle("Italic")


    idx_regular = np.random.choice(np.argwhere(data.context["Italic"].values==0)[:,0],N)
    plt.close("regular")
    plt.figure("regular")
    for i,idx in enumerate(idx_regular):
        plt.subplot(5,5,i+1)
        plt.imshow(data.X[idx,...].squeeze(),cmap='gray',aspect='auto')
    plt.suptitle("Regular")
    