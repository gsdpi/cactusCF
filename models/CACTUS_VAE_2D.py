import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
from address import *
import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers

import json


class CACTUS_VAE_2D(BaseModel):
    def __init__(self,data, params: dict, **kwargs) -> None:      
                
        # Data
        self.params = params
        self.X_train = data.X_train
        self.y_train = data.y_train

        self.training_hist =None
        self.model =None

        self.w,self.h,self.nChannels = self.X_train.shape[1:]

        # Net params        
        self.name              = params.get("name","CACTUS_VAE_2D")
        self.convBlocks        = params.get('convBlocks',2)
        self.ksize             =  params.get('ksize',3)
        self.filters           = params.get('filters',8)
        self.filters_lt        = params.get('filters_lt',26)
        self.class_out_att     = params.get('class_out',["Bold","Italic"])
        self.alpha             = params.get('alpha',0.5)
        self.gamma             = params.get('gamma',2)
        self.initial_epoch_C   = params.get('initial_epoch_C',0) 
        self.final_epoch_C     = params.get('final_epoch_C',10)  
        self.slope_C           = params.get('slope_C',1)  

        self.capacity_callback = capacity_cb(self.slope_C,0,self.initial_epoch_C,self.final_epoch_C)

        # Training params
        self.epochs = params.get('epochs', 2)
        self.patience = params.get('patience', 50)
        self.batch_size = params.get('batch_size',64)
        self.learning_rate = params.get('learning_rate',0.01)
        
        # Initializers
        self.initializer_relu = initialiazers.VarianceScaling(seed=42)
        self.initializer_linear = initialiazers.RandomNormal(0.,0.02,seed=42)
        
        # Metrics
        self.metrics     = []        
        
        # Att output reg
        self.y_class_train = data.context_train[self.class_out_att].values
        
        # Model
        print(f"Building model: {CACTUS_VAE_2D.get_model_name()} [{CACTUS_VAE_2D.get_model_type()}]")
        self.create_model()
        #############################################################################
    def sampling(self,z_m,z_log):        
        batch = tf.shape(z_m)[0]
        dim = tf.shape(z_m)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_m + tf.exp(0.5 * z_log) * epsilon
    

    def _createEnc(self, x):
        self.layersEnc = []
        
        filters = [i*2*self.filters for i in range(self.convBlocks)]; filters[0] = self.filters
        for ll in range(self.convBlocks):
            self.layersEnc.append(layers.Conv2D(kernel_size=self.ksize,
                                                filters=filters[ll],
                                                strides=1,
                                                padding = "same",
                                                activation="relu",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_1")
                                 )
            
            
            self.layersEnc.append(layers.Conv2D(kernel_size=self.ksize,
                                                filters=filters[ll],
                                                strides=1,
                                                padding = "same",
                                                activation="relu",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_2")
                                 )
            self.layersEnc.append(layers.Conv2D(kernel_size=self.ksize,
                                                filters=filters[ll]*2,
                                                strides=2,
                                                padding = "same",
                                                activation="relu",
                                                kernel_initializer= self.initializer_relu,
                                                bias_initializer= self.initializer_relu,
                                                name=f"Enc_conv{ll}_3")
                                 )
        # building enc
        y = x
        for ll,_ in enumerate(self.layersEnc):
            y = self.layersEnc[ll](y)
        return y

    def _createDec(self,z):
        self.layersDec = []
        filters = [i*2*self.filters for i in range(self.convBlocks)]; filters[0] = self.filters
        for ll in reversed(range(self.convBlocks)):
            self.layersDec.append(layers.Conv2DTranspose(kernel_size=self.ksize,
                                                         filters=filters[ll]*2,
                                                         strides=2,
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_1")
                                    )
            self.layersDec.append(layers.Conv2D(kernel_size=self.ksize,
                                                         filters=filters[ll],
                                                         strides=1,
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_2")
                                    )
            self.layersDec.append(layers.Conv2D(kernel_size=self.ksize,
                                                         filters=filters[ll],
                                                         strides=1,
                                                         padding = "same",
                                                         activation="relu",
                                                         kernel_initializer= self.initializer_relu,
                                                         bias_initializer= self.initializer_relu,
                                                         name=f"Dec_conv{ll}_3")
                                    )
        y = z
        for layer in self.layersDec:
            y = layer(y)
        return y

    def create_model(self,verbose=True):
        

        # Input layers
        self.main_input_layer = layers.Input(dtype = tf.float32,shape=[self.w,self.h,1],name='main_input')
        
        # FCN dAE 
        # Encoder 
        self.enc_out = self._createEnc(self.main_input_layer)
        
        # Bottleneck
        self.z_m = layers.Conv2D(kernel_size=int(self.h/(2**self.convBlocks)),
                               filters=self.filters_lt,
                               strides=1,
                               padding = "valid",
                               activation=None,
                               kernel_initializer= self.initializer_linear ,
                               bias_initializer= self.initializer_linear,
                               name="z_m")(self.enc_out)
        
        self.z_log = layers.Conv2D(kernel_size=int(self.h/(2**self.convBlocks)),
                               filters=self.filters_lt,
                               strides=1,
                               padding = "valid",
                               activation=None,
                               kernel_initializer= self.initializer_linear ,
                               bias_initializer= self.initializer_linear,
                               name="z_log")(self.enc_out)
        dims = self.z_m.shape[1:]
        self.z_m = layers.Flatten()(self.z_m)
        self.z_log = layers.Flatten()(self.z_log)

        self.z = self.sampling(self.z_m,self.z_log)
        


        self.z_input = layers.Input(dtype = tf.float32,shape=self.z.shape.as_list()[1:],name='z_input')
        #### CLASSIFICATION #####
        self.z_in_class = layers.Flatten()(self.z_input)
        #self.hidden_class =self.z_in_class
        self.hidden_class = layers.Dense(units = 20,activation='tanh',name="hidden_class")(self.z_in_class) 
        self.class_out = layers.Dense(units = self.y_class_train.shape[-1],activation='sigmoid',name="out_class")(self.hidden_class)
        
        #### DECODER #####
        self.y = layers.Reshape(dims)(self.z_input)
        self.z_dec_input = layers.Conv2DTranspose(kernel_size=int(self.h/(2**self.convBlocks)),
                                                  filters=self.filters_lt,
                                                  strides=1,
                                                  padding = "valid",
                                                  activation=None,
                                                  kernel_initializer= self.initializer_linear,
                                                  bias_initializer= self.initializer_linear,
                                                  name="D_input")(self.y)                    
        


        

        self.y = self._createDec(self.z_dec_input) 

        #Output
        self.y = layers.Conv2D(kernel_size=1,
                               filters=self.nChannels,
                               strides=1,
                               padding = "same",
                               activation='sigmoid',
                                kernel_initializer= self.initializer_linear,bias_initializer= self.initializer_linear,name="out_conv")(self.y)
        

        
        self.model_enc = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.z])
        self.model_dec = tf.keras.Model(inputs=[self.z_input],outputs = [self.y])
        self.classifier = tf.keras.Model(inputs=[self.z_input],outputs = [self.class_out])
        

        self.model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.model_dec(self.model_enc(self.main_input_layer)),self.classifier(self.model_enc(self.main_input_layer))])

        self.model_training = BetaVAEModelTrainStep(inputs= self.main_input_layer,
                                           outputs=[self.model_dec(self.model_enc(self.main_input_layer)),
                                                    self.z_m,
                                                    self.z_log,
                                                    self.classifier(self.model_enc(self.main_input_layer))],
                                           gamma = self.gamma,
                                           alpha = self.alpha 
                                           )
 
        self.model_training.compile( optimizer='adam')
        
        if verbose:
            self.model_training.summary()

        return None
    
    def train(self):
        print(f"Training model: {CACTUS_VAE_2D.get_model_name()} [{CACTUS_VAE_2D.get_model_type()}]")
        
        train_X, v_X, train_class,v_class =  train_test_split(self.X_train, self.y_class_train, test_size=.15,random_state=10)  
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist=self.model_training.fit([train_X], [train_X,train_class] ,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=([v_X], [v_X,v_class]),
                        callbacks=[self.capacity_callback])
    
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

        print(f"saving weights of model {CACTUS_VAE_2D.get_model_name()}: {self.name}")
        self.model.save_weights(savePath+f"/{self.name}")


        self.training_data.to_csv(os.path.join(savePath,"training_data.csv"))
        return None
    
    def load(self):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
    
        loadPath = os.path.join(path_weights, f"{self.name}")
        print(f"restoring weights of model {CACTUS_VAE_2D.get_model_name()}: {self.name}")
        self.model.load_weights(loadPath+f"/{self.name}")
        self.training_data = pd.read_csv(os.path.join(loadPath,"training_data.csv"))
        return None
    
    @classmethod
    def get_model_type(cls):
        return "gen" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CACTUS_VAE_2D" # Aquí se puede indicar un ID que identifique el modelo
    

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 




class BetaVAEModelTrainStep(tf.keras.Model):
    def __init__(self,alpha,gamma,**kwargs):
        super(BetaVAEModelTrainStep, self).__init__(**kwargs) 
        # Tensor for the capacity
        self.Capa = tf.Variable(0,dtype='float32',trainable=False)
        self.gamma = gamma
        self.alpha = alpha
        
        
    def train_step(self, data):
        _,(x,y)=data
        sample_weight = None
        with tf.GradientTape() as tape:
            tape.watch(self.Capa)
            x_pred,z_m,z_log,y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            kl_loss = -0.5 * (1 + z_log - K.square(z_m) - K.exp(z_log))
            kl_loss = K.sum(kl_loss, axis=1, keepdims=True)
            kl_loss = K.mean(kl_loss)
            kl_loss =self.gamma* K.abs(kl_loss - self.Capa)

            reco_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x,x_pred))
            
            cls_loss = tf.keras.losses.BinaryCrossentropy()(y,y_pred)

            loss = self.alpha*(reco_loss + kl_loss) + (1-self.alpha)*cls_loss
            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": loss, "reco_loss": reco_loss,"kl_loss": kl_loss,"cls_loss":cls_loss, "beta":self.Capa}

    def test_step(self, validation_data):        
        _,(x,y) = validation_data
        x_pred,z_m,z_log,y_pred = self(x, training=False)
        kl_loss_val = -0.5 * (1 + z_log - tf.square(z_m) - tf.exp(z_log))
        kl_loss_val = tf.reduce_mean(tf.reduce_sum(kl_loss_val,axis=-1)) 
        reco_loss_val = tf.reduce_mean(tf.keras.losses.mean_squared_error(x,x_pred))
        cls_loss_val = tf.keras.losses.BinaryCrossentropy()(y,y_pred)
        loss_val = self.alpha*(reco_loss_val + kl_loss_val) + (1-self.alpha)*cls_loss_val

        return {"loss": loss_val} # just the reconstruction



# Class to control the capacity of beta-VAE  during training
class capacity_cb(tf.keras.callbacks.Callback):
    def __init__(self, slope:float,capacity_0:float, start_epoch_capacity:int,end_epoch_capacity:int):
        """
        PARAMS:
            slope (float):              rate of capacity increase
            capacity_0 (float):         capacity for epoch 0
            start_epoch_capacity (int): epoch from which the capacity will be increased
            end_epoch_capacity (int):   epoch from which the the increment of the capacity will stop

        OUTPUT
            Keras callback that gradually increase the value of capacity stored in a TF variable. 
        """
        super(capacity_cb).__init__()
        self.capacity_0=capacity_0
        self.slope = slope
        self.val_capacity = capacity_0
        self.start_epoch = start_epoch_capacity
        self.end_epoch = end_epoch_capacity
        
        
    def on_epoch_end(self,epoch, logs=None):
        if (epoch >= self.start_epoch) and (epoch <= self.end_epoch):
            self.val_capacity = max(0,epoch*self.slope + self.capacity_0)
            
        # The variable must be defined in the model. 
        K.set_value(self.model.Capa, tf.constant(self.val_capacity))
        #TODO: This callback must be coordinated with the model in order to avoid future inconsistent errors beacuse
        #      of the capacity variable's name

##########################################
# Unit testing
##########################################


if __name__ == "__main__":
    pass