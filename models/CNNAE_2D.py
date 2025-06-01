import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
from address import *

layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers



class CNNAE_2D(BaseModel):
    def __init__(self,data, params: dict, **kwargs) -> None:      
                
        # Data
        self.params = params
        self.X_train = data.X_train
        self.y_train = data.y_train

        self.training_hist =None
        self.model =None

        self.w,self.h,self.nChannels = self.X_train.shape[1:]

        # Net params        
        self.name    = params.get("name","CNNAE_2D")

        # Training params
        self.epochs = params.get('epochs', 2)
        self.patience = params.get('patience', 20)
        self.batch_size = params.get('batch_size',64)
        
        
        # Metrics
        self.metrics     = []        
        
        # Model
        print(f"Building model: {CNNAE_2D.get_model_name()} [{CNNAE_2D.get_model_type()}]")
        self.create_model()
        #############################################################################
        
    def create_model(self,verbose=True):
        

        # Input layers
        self.main_input_layer = layers.Input(dtype = tf.float32,shape=[self.w,self.h,1],name='main_input')
        
        # Encoder 
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.main_input_layer)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        self.encoded = layers.Conv2D(1, (3, 3), activation=None, padding='same')(x)
    
        # decoder        
        self.z_input = layers.Input(dtype = tf.float32,shape=self.encoded.shape.as_list()[1:],name='dec_input')

        
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.z_input)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        self.decoded = layers.Conv2D(1, (3, 3), activation=None, padding='same')(x)
        
        
        self.model_enc = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.encoded])
        self.model_dec = tf.keras.Model(inputs=[self.z_input],outputs = [self.decoded])
        

        self.model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.model_dec(self.model_enc(self.main_input_layer))])



        self.model.compile( optimizer='adam', loss="mse")
        
        if verbose:
            self.model.summary()

        return None
    
    def train(self):
        print(f"Training model: {CNNAE_2D.get_model_name()} [{CNNAE_2D.get_model_type()}]")
        
        train_X, v_X =  train_test_split(self.X_train, test_size=.15,random_state=10)  
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist=self.model.fit([train_X], [train_X] ,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=([v_X], [v_X]),
                        callbacks=[ES_cb])
    
        self.training_data = pd.DataFrame(self.training_hist.history)
        return self.training_hist        

    def predict(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample 
            RETURN
                X_ [numpy array] -> Estimated input sample
        """

        o,_ = self.model.predict(X)
        return o
  

    def predict_enc(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample 
            RETURN
                z[numpy array] -> Bottleneck
        """


        return self.model_enc.predict(X)

    def evaluate(self, X,metrics):

        return None

    def store(self):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        
        savePath = os.path.join(path_weights, f"{self.name}")
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        print(f"saving weights of model {CNNAE_2D.get_model_name()}: {self.name}")
        self.model.save_weights(savePath+f"/{self.name}")


        self.training_data.to_csv(os.path.join(savePath,"training_data.csv"))
        return None
    
    def load(self):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
    
        loadPath = os.path.join(path_weights, f"{self.name}")
        print(f"restoring weights of model {CNNAE_2D.get_model_name()}: {self.name}")
        self.model.load_weights(loadPath+f"/{self.name}")
        self.training_data = pd.read_csv(os.path.join(loadPath,"training_data.csv"))
        return None
    
    @classmethod
    def get_model_type(cls):
        return "gen" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CNNAE_2D" # Aquí se puede indicar un ID que identifique el modelo
    

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 



##########################################
# Unit testing
##########################################


if __name__ == "__main__":
    pass