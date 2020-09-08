from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as  pd
from sklearn import datasets
from keras import backend as k
from sklearn import decomposition
from decimal import Decimal

timesteps = 3   # seq length
input_dim = 25 # number of exec features

n_features = 25


#features_fileName1 = "bayes_features_bigdata.csv"
features_fileName1 = "../../dataset//bayes_features_q_BD_4nodes.csv"
features_fileName2 = "../../dataset//bayes_features_half_bigdata_4nodes.csv"
features_fileName3 = "../../dataset//bayes_features_bigdata_4nodes.csv"
features_fileName4 = "../../dataset//bayes_features_2xbigdata_4nodes.csv"
features_fileName5= "../../dataset//bayes_features_3xBD_4nodes.csv"

features_fileName6 = "../../dataset//pr_features_5M_4nodes.csv"
features_fileName7 = "../../dataset//pr_features_10M_4nodes.csv"
features_fileName8 = "../../dataset//pr_features_15M_4nodes.csv"
features_fileName9 = "../../dataset//pr_features_20m_4nodes.csv"
features_fileName10 = "../../dataset//pr_features_25m_4nodes.csv"


features_fileName11= "../../dataset//wc_features_ds1_4nodes.csv"
features_fileName12 = "../../dataset//wc_features_gigantic_4nodes.csv"
features_fileName13= "../../dataset//wc_features_half_BD_4nodes.csv"
features_fileName14 = "../../dataset//wc_features_ds2_4nodes.csv"
features_fileName15 = "../../dataset//wc_features_BD_4nodes.csv"


features_fileName16 = "../../dataset//tpch_featues_4nodes_20.csv"
features_fileName17 = "../../dataset//tpch_featues_4nodes_40.csv"
features_fileName18 = "../../dataset//tpch_featues_4nodes_60.csv"
features_fileName19 = "../../dataset//tpch_featues_4nodes_80.csv"
features_fileName20 = "../../dataset//tpch_featues_4nodes_100.csv"



features_fileName21 = "../../dataset//terasort_featues_4nodes_ds1.csv"
features_fileName22 = "../../dataset//terasort_featues_4nodes_ds2.csv"
features_fileName23 = "../../dataset//terasort_featues_4nodes_ds3.csv"
features_fileName24 = "../../dataset//terasort_featues_4nodes_ds4.csv"
features_fileName25 = "../../dataset//terasort_featues_4nodes_ds5.csv"

gcp_features_fn = "../../dataset//exec-metrics-gcp-20nodes.csv"


#load features data
def load_data(fn):
  with open(fn) as csvfile:
    readerCSV = csv.reader(csvfile, delimiter=',')
    header = next (readerCSV)
    header = header [:24]
    print (">>> header >>> " , header)
    features = []
    target = []
    for r in readerCSV:
        features = np.append( features , r[0:24])
        target = np.append (target , r [25])
#       features = np.append( features , target)  ## add the target to the features for encoding
#print (">>>> length >>> ", len (features))
# target = target.astype (float)
# target = target [:10]
##  features = features [:int(len(features)/24)*24]
  features = np.reshape (features, (int(len (features)/24),24))

  features, v_features, target, y_test = train_test_split(features, target, test_size=0.2)

  return target , features , v_features

  
def load_test_data(fn):
  with open(fn) as csvfile:
    readerCSV = csv.reader(csvfile, delimiter=',')
    header = next (readerCSV)
    header = header [:24]
    print (">>> header >>> " , header)
    features = []
    target = []
    for r in readerCSV:
        features = np.append( features , r[0:24])
        
  features = np.reshape (features, (int(len (features)/24),24))
  return  features 
  
  
scaler_filename = "scaler.save"
    
### scale the features
##########
def scale_data (features): 
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.externals import joblib
  scaler = MinMaxScaler ()
  scaler = StandardScaler()
  scaler = MaxAbsScaler()
  scaled_df =  scaler.fit_transform (features )
  joblib.dump(scaler, scaler_filename)
  return scaled_df

def scale_val_data (features): 
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.externals import joblib
  scaler = joblib.load(scaler_filename) 
  scaled_df =  scaler.fit_transform (features )
  joblib.dump(scaler, scaler_filename)
  return scaled_df
  

target,   features  ,   v_features = load_data (features_fileName1)
target2,  features2 ,   v_features2= load_data (features_fileName2)
target3,  features3 ,   v_features3= load_data (features_fileName3)
target4,  features4 ,   v_features4= load_data (features_fileName4)
target5, features5 ,   v_features5= load_data (features_fileName5)
target6,  features6 ,   v_features6= load_data (features_fileName6)
target7,  features7,   v_features7 = load_data (features_fileName7)
target8,  features8 ,   v_features8= load_data (features_fileName8)
target9,  features9 ,   v_features9= load_data (features_fileName9)
target10,  features10,   v_features10 = load_data (features_fileName10)
target11,  features11,   v_features11 = load_data (features_fileName11)
target12,  features12,   v_features12 = load_data (features_fileName12)
target13,  features13,   v_features13 = load_data (features_fileName13)
target14,  features14,   v_features14 = load_data (features_fileName14)
target15,  features15 ,   v_features15= load_data (features_fileName15)
target16,  features16 ,   v_features16= load_data (features_fileName16)
target17,  features17,   v_features17 = load_data (features_fileName17)
target18,  features18,   v_features18 = load_data (features_fileName18)
target19,  features19,   v_features19 = load_data (features_fileName19)
target20,  features20 ,   v_features20= load_data (features_fileName20)


target21,  features21 ,   v_features21= load_data (features_fileName21)
target22,  features22,   v_features22 = load_data (features_fileName22)
target23,  features23,   v_features23 = load_data (features_fileName23)
target24,  features24,   v_features24 = load_data (features_fileName24)
target25,  features25 ,   v_features25= load_data (features_fileName25)








print (">> val features shape >> " , v_features.shape)
print (">>  features shape >> " , features.shape)

target = np.append (target , target2)
target = np.append (target , target3)
target = np.append (target , target4)
target = np.append (target , target5)
target = np.append (target , target6)
target = np.append (target , target7)
target = np.append (target , target8)
target = np.append (target , target9)
target = np.append (target , target10)
target = np.append (target , target11)
target = np.append (target , target12)
target = np.append (target , target13)
target = np.append (target , target14)
target = np.append (target , target15)
target = np.append (target , target16)
target = np.append (target , target17)
target = np.append (target , target18)
target = np.append (target , target19)
target = np.append (target , target20)
target = np.append (target , target21)
target = np.append (target , target22)
target = np.append (target , target23)
target = np.append (target , target24)
target = np.append (target , target25)




features = np.append (features , features2)
features = np.append (features , features3)
features = np.append (features , features4)
features = np.append (features , features5)

features = np.append (features , features6)
features = np.append (features , features7)
features = np.append (features , features8)
features = np.append (features , features9)
features = np.append (features , features10)
features = np.append (features , features11)
features = np.append (features , features12)
features = np.append (features , features13)
features = np.append (features , features14)
features = np.append (features , features15)
features = np.append (features , features16)
features = np.append (features , features17)
features = np.append (features , features18)
features = np.append (features , features19)
features = np.append (features , features20)

features = np.append (features , features21)
features = np.append (features , features22)
features = np.append (features , features23)
features = np.append (features , features24)
features = np.append (features , features25)

features = np.reshape (features,((int (len (features)/24)),24))


v_features = np.append (v_features , v_features2)
v_features = np.append (v_features , v_features3)
v_features = np.append (v_features , v_features4)
v_features = np.append (v_features , v_features5)

v_features = np.append (v_features , v_features6)
v_features = np.append (v_features , v_features7)
v_features = np.append (v_features , v_features8)
v_features = np.append (v_features , v_features9)
v_features = np.append (v_features , v_features10)

v_features = np.append (v_features , v_features11)
v_features = np.append (v_features , v_features12)
v_features = np.append (v_features , v_features13)
v_features = np.append (v_features , v_features14)
v_features = np.append (v_features , v_features15)


v_features = np.append (v_features , v_features16)
v_features = np.append (v_features , v_features17)
v_features = np.append (v_features , v_features18)
v_features = np.append (v_features , v_features19)
v_features = np.append (v_features , v_features20)

v_features = np.append (v_features , v_features21)
v_features = np.append (v_features , v_features22)
v_features = np.append (v_features , v_features23)
v_features = np.append (v_features , v_features24)
v_features = np.append (v_features , v_features25)

v_features = np.reshape (v_features,((int (len (v_features)/24)),24))

features = scale_data (features)
X = features
v_features = scale_val_data (v_features)
test_features = load_test_data(gcp_features_fn)
test_features = scale_val_data(test_features)
encoding_dim = 5
n_exp = 10
initial_seed = 1337 
######### pca loss per component ######
'''
from sklearn import decomposition
for i in range (n_exp):
    pca_component_loss = []
    exp_variance = []
    seed = initial_seed *i
    np.random.seed(seed)
    pca = decomposition.PCA (n_components=encoding_dim)
    pca.fit(X)
    pca_x = pca.transform (X)
    X_projected = pca.inverse_transform(pca_x)

    pca_val_x = pca.transform (test_features)
    X_test_projected = pca.inverse_transform(pca_val_x)
    pca_loss = (( X_test_projected-test_features) ** 2).mean()

    exp_variance = pca.explained_variance_ratio_.cumsum()
    #print (">>> var >>> " + str (exp_variance))
    writer  = open ( "/local/data/experiment/pca_test_loss_5components.csv" , 'a')
    #for i in range (len (pca_loss)):
    writer.write (str(pca_loss) + "," + str(exp_variance)+"\n")
    writer.close ()
'''
########################################################
########################################################
#learn the encoding or load the encoded learnt model
from keras.layers import Input, Dense
from keras.models import Model


input_dim = 24


#from keras.models import load_model
#model = load_model('model.h5')

n_epoch = 300
#######################################################

for i in range (n_exp):
    seed = initial_seed *(i+1)
    np.random.seed(seed)
    input_features = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='linear')(input_features)
    linear_encoder = Model(input_features, encoded)  ### encoder model
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(input_dim, activation='linear')(encoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_features, decoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    hist = autoencoder.fit(X, X,
                    epochs=300,
                    batch_size=32 , verbose=1 , shuffle =1 , validation_data=(test_features, test_features))

    linear_ae_loss = hist.history['val_loss']
    linear_encoder.save("linear_encoder_model_300.h5")
    autoencoder.save("linear_AE__model_300.h5")
    ######


    input_features_2 = Input(shape=(input_dim,))

    # "encoded" is the encoded representation of the input
    nonlinear_encoded = Dense(encoding_dim, activation='sigmoid')(input_features_2)
    # "decoded" is the lossy reconstruction of the input
    nonlinear_encoder = Model(input_features_2, nonlinear_encoded)  ### encoder model
    nonlinear_decoded = Dense(input_dim, activation='sigmoid')(nonlinear_encoded)

    nonlinear_autoencoder = Model(input_features_2, nonlinear_decoded)

    nonlinear_autoencoder.compile(optimizer='adam', loss='mse')

    hist = nonlinear_autoencoder.fit(X, X,
                    epochs=300,
                    batch_size=32 , verbose=1 , shuffle =1, validation_data=(test_features, test_features))

    nonlinear_ae_loss = hist.history['val_loss']
    nonlinear_encoder.save("nonlinear_encoder_model_300.h5")
    nonlinear_autoencoder.save("nonlinear_AE__model_300.h5")

    
    writer  = open ( "/local/data/experiment/ae_test_loss_5components_GCP_300.csv" , 'a')
    for i in range (len (linear_ae_loss)):
          writer.write(str(linear_ae_loss[i]) + "," + str(nonlinear_ae_loss[i])+"\n")
        
    writer.close()
######################################

#'''







