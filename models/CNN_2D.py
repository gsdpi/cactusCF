from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
from address import *
import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC,CategoricalAccuracy
import numpy as np
import pandas as pd
import json



class CNN_2D(BaseModel):

    def __init__(self, data, params: dict, **kwargs)->None:
        
        # Data
        self.params = params
        self.X_train = data.X_train
        self.y_train = data.y_train

        self.training_hist =None
        self.model =None

        # Net params
        self.M, self.N          = self.X_train.shape[1:-1]
        self.nChannels          = self.X_train.shape[-1]
        self.cls_labels         = np.unique(self.y_train)
        self.nb_clss            = len(self.cls_labels)
        
        self.name               = params.get("name","2DCNN")
        self.filters            = params.get("filters",[32,64])
        self.units            = params.get("filters",[100,30])
        self.kernel_size        = params.get("kernel_size",3)
        self.ConvBlocks         = len(self.filters)
        self.layers = []

        # Training params
        self.epochs     = params.get('epochs', 2)
        self.patience   = params.get('patience', 15)
        self.batch_size = params.get('batch_size',64)        

        # Metrics
        self.metrics    = [CategoricalAccuracy(), AUC()]

        # Model
        print(f"Building model: {CNN_2D.get_model_name()} [{CNN_2D.get_model_type()}]")
        self.create_model()

    def ConvBlock(self,filters,kernel_size,name):
        self.layers.append(layers.Conv2D(filters=filters, kernel_size= kernel_size, padding='same',activation='relu',name=f"{name}_1"))
        self.layers.append(layers.Conv2D(filters=filters, kernel_size= kernel_size, padding='same',activation='relu',name=f"{name}_2"))
        self.layers.append(layers.Conv2D(filters=filters, kernel_size= kernel_size, padding='same',activation='relu',name=f"{name}_3"))
        self.layers.append(layers.MaxPool2D(pool_size=(2,2)))


    def create_model(self,verbose=True):
        # Input layer
        self.main_input_layer = layers.Input(dtype=tf.float32, shape= [self.M,self.N,self.nChannels],name='main_input')
        # Conv blocks
        for l,filter in enumerate(self.filters):
            self.ConvBlock(filter,self.kernel_size,f"Conv{l}")
        # Flatten
        self.layers.append(layers.Flatten(name="flatten"))
        # Fully-connected layers
        for l,u in enumerate(self.units):
            self.layers.append(layers.Dense(units=u,activation="relu",name=f"FC_{l}"))
        
        # Output layer
        self.layers.append(layers.Dense(self.nb_clss,activation="softmax"))

        self.y = self.main_input_layer
        for layer in self.layers:
            self.y = layer(self.y)

        self.model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.y])
        # Compiling the model 
        self.model.compile( optimizer='adam',
                        loss="categorical_crossentropy",
                        metrics=self.metrics)

        if verbose:
            self.model.summary()

        return None       
    
    
    def train(self):
        print(f"Training model: {CNN_2D.get_model_name()} [{CNN_2D.get_model_type()}]")
        train_X, v_X, train_y, v_y =  train_test_split(self.X_train, self.y_train, test_size=0.15,random_state=10)  
        
        
        
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist=self.model.fit([train_X], [train_y] ,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(v_X, v_y),
                        callbacks=[ES_cb])
        
        self.training_data = pd.DataFrame(self.training_hist.history)
    def predict(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample 
            RETURN
                y_ [numpy array] -> Estiamted class labels
        """
        return self.model.predict(X) 
    
    def store(self):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        
        savePath = os.path.join(path_weights, f"{self.name}")
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        print(f"saving weights of model {CNN_2D.get_model_name()}: {self.name}")
        self.model.save_weights(savePath+f"/{self.name}")

        self.training_data.to_csv(os.path.join(savePath,"training_data.csv"))
        return None
    
    def load(self):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
    
        loadPath = os.path.join(path_weights, f"{self.name}")
        print(f"restoring weights of model {CNN_2D.get_model_name()}: {self.name}")
        self.model.load_weights(loadPath+f"/{self.name}")
        self.training_data = pd.read_csv(os.path.join(loadPath,"training_data.csv"))
        return None
    

    @classmethod
    def get_model_type(cls):
        return "class" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CNN_2D" # Aquí se puede indicar un ID que identifique el modelo

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 