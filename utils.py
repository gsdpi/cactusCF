# from scipy import signal
# from scipy.signal import medfilt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
import gc
from tensorflow.keras import backend as K

import ipdb

def cleanup_gpu():
    # Delete model from memory
    K.clear_session()  # Clear backend session
    tf.keras.backend.clear_session()  # Clear Keras-related memory
    gc.collect()  # Run garbage collection

def upsampling(X,y, strategy="all"):
    oversample = RandomOverSampler(sampling_strategy=strategy)
    y_ = np.argmax(y,axis=1) if len(y.shape) > 1 else y 
    X_ = X[...,1] if len(X.shape)>2 else X
    oversample.fit_resample(X_,y)
    idx = oversample.sample_indices_
    return X[idx,...],y[idx,...], idx



# originally from: https://github.com/zhendong3wang/learning-time-series-counterfactuals/blob/main/src/help_functions.py
def euclidean_distance(X, cf_samples, average=True):
    X           = np.reshape(X,(X.shape[0],-1))
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))

    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances) if average else paired_distances

def norm_euclidean_distance(X, cf_samples, average=True):
    X           = np.reshape(X,(X.shape[0],-1))
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))


    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances)/np.sqrt(cf_samples.shape[0]) if average else paired_distances


# originally from: https://github.com/zhendong3wang/learning-time-series-counterfactuals/blob/main/src/help_functions.py
def validity_score(pred_labels, cf_labels,accuracy_score):
    desired_labels = 1 - pred_labels  # for binary classification
    return accuracy_score(y_true=desired_labels, y_pred=cf_labels)

# LOF of conunterfactuals on the true data distribution
def LOF_score(X_train, cf_samples):
    # Flattening the X and Xcf 
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))
    X_train     = np.reshape(X_train,(X_train.shape[0],-1))
    LOF_model =  LocalOutlierFactor(n_neighbors=int(np.sqrt(X_train.shape[0])),novelty=True)
    LOF_model.fit(X_train)
    LOF = LOF_model.score_samples(cf_samples)
    return np.mean(-LOF.squeeze())

# LOF of conunterfactuals on the true context distribution
def LOF_context_score(X_train,context_train, cf_samples,target_context,agg = True):
    # Flattening the X and Xcf 
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))
    X_train     = np.reshape(X_train,(X_train.shape[0],-1))

    LOFs = []
    LOFs_labels = np.unique(context_train,axis=0)
    # For each context class a LOF model is trained and all the Xcf belonging to this class are evaluated
    for context in LOFs_labels:
        context = context.tolist()

        #LOF value is only computed if there is samples of this context both in training dataset and in the generated Xcf
        if (np.sum(np.all(target_context == context, axis=1)) > 0)  and (np.sum(np.all(context_train == context, axis=1)) > 20):
           
            # filtering X cf
            idx = np.all(target_context == context, axis=1)
            idx = np.where(idx)[0]
            cf_samples_ = cf_samples[idx]
                        
            # filtering X train
            idx = np.all(context_train == context, axis=1)
            idx = np.where(idx)[0]
            X_ = X_train[idx]

            #LOF_model =  LocalOutlierFactor(n_neighbors=int(np.sqrt(len(idx))),novelty=True)
            LOF_model =  LocalOutlierFactor(n_neighbors=10,novelty=True,metric="manhattan")
            LOF_model.fit(X_) # It returns a number. Large negative values means outlier; values close to 0 means inliers

            LOFs.append(-LOF_model.score_samples(cf_samples_))

    LOFs = np.hstack(LOFs).squeeze()
    if agg:
        return np.mean(LOFs)
    else:
        return LOFs
    
    



# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples):
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    X           = np.reshape(X,(X.shape[0],-1))
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))
    c = np.isclose(X, cf_samples, atol=0.01) 
    # return a positive compactness, instead of 1 - np.mean(..)
    return np.mean(c)


def cf_eval(X_train,context_train,X,CF_X,pred_labels, target_context, CF_labels,accuracy_score=f1_score):

    dist        = euclidean_distance(X,CF_X)
    dist_norm   = norm_euclidean_distance(X,CF_X)
    validity    = validity_score(pred_labels,CF_labels,accuracy_score)
    compactness = compactness_score(X,CF_X)
    lof_context = LOF_context_score(X_train, context_train, CF_X,target_context)
    #lof         = LOF_score(X_train, CF_X)  
    return [dist,dist_norm,validity,compactness,lof_context],["proximity","n_proximity","validity","compactness","lof_context"]



def pairwise_l2_norm2(x, y, scope=None):
    
    size_x = tf.shape(x).numpy()[0]
    size_y = tf.shape(y).numpy()[0]
    xx = tf.expand_dims(x, -1)
    xx = tf.tile(xx, tf.constant([1, 1, 1,size_y]))

    yy = tf.expand_dims(y, -1)
    yy = tf.tile(yy, tf.constant([1, 1,1, size_x]))
    yy = tf.transpose(yy, perm=[2, 1, 0])

    diff = tf.math.subtract(xx, yy)
    square_diff = tf.square(diff)

    square_dist = tf.reduce_sum(square_diff, 1)

    return square_dist


def polynomial_decay(initial_lr, end_lr, max_epochs, power, current_epoch):
    """
    Polynomial learning rate decay.

    Parameters:
        initial_lr (float): Initial learning rate
        end_lr (float): Final learning rate after decay
        max_epochs (int): Total number of epochs
        power (float): Polynomial power (1.0 = linear decay)
        current_epoch (int): Current epoch

    Returns:
        float: Adjusted learning rate
    """
    if current_epoch > max_epochs:
        return end_lr
    decay = (1 - current_epoch / max_epochs) ** power
    lr = (initial_lr - end_lr) * decay + end_lr
    return lr


def styling_mpl_table(table,table_data,fontsize = 12):
    """
        Customize the appearance: remove vertical lines and lighten horizontal lines
    """
    
    colLabels = [cell.get_text().get_text() for (row, col), cell in table.get_celld().items() if row==0]
    n_cols = len(colLabels)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)  # lighter lines
        cell.set_edgecolor('lightgray')  # lighter color
        cell.set_height(0.1) 
        # Remove vertical borders inside the table
        cell.visible_edges = 'horizontal'

        # Remove the first horizontal line
        if row ==0:
            cell.visible_edges = 'open'
     # Format of the table
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for i, col in enumerate(colLabels):
        max_len = max([len(str(row[i])) if row[i] is not None else 0 for row in table_data] + [len(col)])
        scale = max_len / 8  
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(scale * 0.1)  # Escala proporcional al contenido

    # Optional: style header separately
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_fontsize(fontsize)
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('gray')
   
                  
