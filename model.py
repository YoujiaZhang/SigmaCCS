from spektral.layers import GlobalSumPool,ECCConv
from spektral.data import BatchLoader

from parameter import *
from GraphData import *

from tensorflow.keras.models import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.metrics import r2_score
import numpy as np
import pickle
np.set_printoptions(suppress=True)

def Bais(y,r):
    '''
    * The gap between the predicted result and the real value of the evaluation model
    *
    * Attributes
    * ----------
    * y : y_true
    * r : y_pred
    '''
    RelativeError = [abs(y[i]-r[i])/y[i] for i in range(len(y))]
    R2_Score = r2_score(r,y)
    # print("R2 Score :", R2_Score, '\n')
    # print("Median Relative Error :", np.median(RelativeError) * 100, '%')
    # print("Mean Relative Error :", np.mean(RelativeError) * 100, '%')
    return R2_Score, np.median(RelativeError) * 100

def one_of_k_encoding_unk(x, allowable_set):
    '''
    * One-hot encoding
    *
    * Attributes
    * ----------
    * x             : Data
    * allowable_set : Data range
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def load_Model_from_file(ModelfilePath):
    Model = load_model(ModelfilePath,custom_objects = {"ECCConv": ECCConv,"GlobalSumPool": GlobalSumPool})
    return Model

def predict(Model,adduct_SET,dataset,adduct):
    '''
    * Predicting CCS with model
    *
    * Attributes
    * ----------
    * Model      : SiGMA model
    * adduct_SET : Adduct Set
    * dataset    : Input Graph data of the model
    * adduct     : Input adduct data of the model
    * 
    * Returns
    * -------
    * y_pred : CCS prediction results of the model
    '''
    # This loader returns batches of graphs stacked along an extra dimension, with all "node" dimensions padded to be equal among all graphs.
    # If n_max is the number of nodes of the biggest graph in the batch, then the padding consist of adding zeros to the node features, adjacency matrix, 
    # and edge attributes of each graph so that they have shapes (n_max, n_node_features), (n_max, n_max), and (n_max, n_max, n_edge_features) respectively.

    # The zero-padding is done batch-wise, which saves up memory at the cost of more computation. If latency is an issue but memory isn't, or if the dataset has graphs with a 
    # similar number of nodes, you can use the PackedBatchLoader that first zero-pads all the dataset and then iterates over it.

    # Note that the adjacency matrix and edge attributes are returned as dense arrays (mostly due to the lack of support for sparse tensor operations for rank >2).
    # Only graph-level labels are supported with this loader (i.e., labels are not zero-padded because they are assumed to have no "node" dimensions).
    
    ## dataset: a graph Dataset;
    ## batch_size: size of the mini-batches;
    ## epochs: number of epochs to iterate over the dataset. By default (None) iterates indefinitely;
    ## shuffle: whether to shuffle the data at the start of each epoch.
    
    loader = BatchLoader(dataset,batch_size=1,epochs=1,shuffle=False)
    loader_data = ()
    ltd_index = 0
    
    # The Graph data of each molecule is spliced with adduct data
    for i in loader.load():
        adduct_one_hot = [one_of_k_encoding_unk(adduct[ltd_index+ltd_index_i],adduct_SET) for ltd_index_i in range(len(i[1]))]
        adduct_one_hot = np.array(adduct_one_hot)
        one_sample = ((adduct_one_hot,i[0][0],i[0][1],i[0][2]),i[1])
        loader_data += (one_sample,)
        ltd_index += len(i[1])
    loader_data = (i for i in loader_data)
    
    y_true = []
    y_pred = []
    count = 0
    for batch in loader_data:
        inputs, target = batch
        predictions = Model(inputs, training=False) # predict
        pred = np.array(predictions[0])
        # print(pred[0],target[0],(abs(pred[0]-target[0])/target[0]*100))
        # print((abs(pred[0]-target[0])/target[0]))
        y_pred.append(pred[0])
        y_true.append(target[0])
        count += 1
    return y_pred

def Mymodel(DataSet,adduct_SET):
    '''
    * Constructing SiGMA model
    *
    * Attributes
    * ----------
    * DataSet    : Input Graph data of the model
    * 
    * Returns
    * -------
    * model : The constructed SiGMA model
    '''
    Kernel_Network = [64,64,64,64]
    F = DataSet.n_node_features
    E = DataSet.n_edge_features
    X_in = Input(shape=(None,F))
    A_in = Input(shape=(None,None))
    E_in = Input(shape=(None,None,E))
    Z_in = Input(shape=(len(adduct_SET),))
    
    ###################################################################################################################
    # ECCConv(channels, kernel_network=None, root=True, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
    #         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    #         kernel_constraint=None, bias_constraint=None)
    ####################################################################################################################
    ## An edge-conditioned convolutional layer (ECC) from the paper
    ## Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs Martin Simonovsky and Nikos Komodakis\
    
    ## Node features of shape ([batch], n_nodes, n_node_features);
    ## Binary adjacency matrices of shape ([batch], n_nodes, n_nodes);
    ## Edge features. In single mode, shape (num_edges, n_edge_features); in batch mode, shape (batch, n_nodes, n_nodes, n_edge_features).
    
    ## channels: integer, number of output channels;
    ## kernel_network: a list of integers representing the hidden neurons of the kernel-generating network;
    ## activation: activation function;
    ## use_bias: bool, add a bias vector to the output;
    ## kernel_initializer: initializer for the weights;
    ## bias_initializer: initializer for the bias vector;
    ## kernel_regularizer: regularization applied to the weights;
    ## bias_regularizer: regularization applied to the bias vector;
    ## activity_regularizer: regularization applied to the output;
    ## kernel_constraint: constraint applied to the weights;
    ## bias_constraint: constraint applied to the bias vector.
    output = ECCConv(16,Kernel_Network,activation="relu",kernel_regularizer='l2')([X_in,  A_in,E_in])
    output = ECCConv(16,Kernel_Network,activation="relu",kernel_regularizer='l2')([output,A_in,E_in])
    output = ECCConv(16,Kernel_Network,activation="relu",kernel_regularizer='l2')([output,A_in,E_in])
    # A global sum pooling layer. Pools a graph by computing the sum of its node features.
    molecule = GlobalSumPool()(output)
    concat = concatenate([Z_in,molecule])
    Dense1 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense2 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense3 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense4 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense5 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense6 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense7 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense8 = Dense(384,activation="relu",kernel_regularizer='l2')
    Dense9 = Dense(1,  activation="relu",)
    DenseS = Dense9(Dense8(Dense7(Dense6(Dense5(Dense4(Dense3(Dense2(Dense1(concat)))))))))
    model = Model(inputs = [Z_in,X_in,A_in,E_in], outputs=DenseS)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model

def train(Model,dataset_tr,adduct_tr,adduct_SET,EPOCHS,BATCHS,Vis):
    '''
    * Training model
    *
    * Attributes
    * ----------
    * Model      : The constructed SiGMA model
    * dataset_tr : Input Graph  data for training
    * adduct_tr  : Input Adduct data for training
    * adduct_SET : Adduct Set
    * EPOCHS     : Number of epochs to iterate over the dataset. By default (None) iterates indefinitely
    * BATCHS     : Size of the mini-batches
    * Vis        : Trained model
    * 
    * Returns
    * -------
    * Model : The constructed SiGMA model
    '''
    for epoch in range(EPOCHS):
        # Random scrambling of data
        dataset_tr_idxs = np.random.permutation(len(dataset_tr))
        dataset_tr_2 = dataset_tr[dataset_tr_idxs]
        adduct_tr_2  = list(np.array(adduct_tr)[dataset_tr_idxs])
        # Loading data
        loader_tr = BatchLoader(dataset_tr_2,batch_size=BATCHS,epochs=1,shuffle=False)
        loader_tr_data = ()
        ltd_index = 0
        # The Graph data of each molecule is spliced with adduct data
        for i in loader_tr.load():
            adduct_one_hot=[one_of_k_encoding_unk(adduct_tr_2[ltd_index+ltd_index_i],adduct_SET) for ltd_index_i in range(len(i[1]))]
            adduct_one_hot = np.array(adduct_one_hot)
            one_sample = ((adduct_one_hot,i[0][0],i[0][1],i[0][2]),i[1])
            loader_tr_data += (one_sample,)
            ltd_index += len(i[1])
        loader_tr_data = (i for i in loader_tr_data)
        # Train
        re = Model.fit(loader_tr_data, steps_per_epoch=loader_tr.steps_per_epoch, epochs=1, verbose=Vis,)
    return Model
