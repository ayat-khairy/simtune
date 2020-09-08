from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.use('Agg')
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
#latent_dim = 4 # compressed representation length
n_features = 25
n_val = 1 #  number of WL executions for fingerprinting

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

  features = features [:int(len(features)/24)*24]
  features = np.reshape (features, (int(len (features)/24),24))

  
  v_features = features [ 0:n_val, :]
  features = features  [ n_val:, :]
  target = target [n_val:]
  

  return target , features , v_features

  
scaler_filename = "scaler.save"
    
### scale the features
##########
def scale_data (features): 
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.externals import joblib
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


v_features = scale_val_data (v_features)


#################calculate distance###############################
def calculate_manhatten_distance (X1,X2):
   distance = [0]*100
   for i in range (len (X2)):
   	for j in range (len (X2[i])):
           	distance[i] += np.abs (X1[i][j]-X2[i][j])   ## manhatten distance
   return distance
#################calculate distance###############################
def calculate_eculedian_distance (X1,X2):
   distance = [0]*100
   for i in range (min (len (X1) , len (X2))):
   	for j in range (len (X1[i])):
           	distance[i] += np.sqrt((X1[i][j]-X2[i][j])**2)   ## manhatten distance
   return distance
##########################################################
def calculate_distance_score (X1 , X2):
 	distance = calculate_manhatten_distance (X1 , X2)
 	distance_score  = np.mean (distance)
 	return distance_score
########################################################
# load the encoded learnt model
from keras.layers import Input, Dense
from keras.models import Model

from keras.models import load_model
encoder = load_model('nonlinear_encoder_model_300.h5')

encoded_v_features = encoder.predict( v_features)

encoding_dim = 5
input_dim = 24

print (encoded_v_features)


wl_names = ["Bayes-DS1" , "Bayes-DS2" ,"Bayes-DS3" , "Bayes-DS4" ,"Bayes-DS5" ,"PR-5m" , "PR-10m" ,"PR-15m" ,"PR-20m" ,"PR-25m" ,"WC-DS1" ,"WC-DS2" , "WC-DS3" , "WC-DS4" , "WC-DS5"  , "TPCH-20" , "TPCH-40" , "TPCH-60" , "TPCH-80" , "TPCH-100" , "Terasort-20", "Terasort-40", "Terasort-60", "Terasort-80", "Terasort-100"]


features_to_compare = np.reshape (encoded_v_features , (25,n_val,encoding_dim))   ## 25 n_files
#features_to_compare = np.reshape (v_features , (25,n_val,24))  ## wzout encoding
distance_matrix= []
for x in features_to_compare:
	for i in range (len (features_to_compare)):
		distance_matrix = np.append (distance_matrix , calculate_distance_score (x , features_to_compare[i]))

distance_matrix = np.reshape (distance_matrix , (len(features_to_compare) ,len(features_to_compare) ))
#distance_matrix = np.flip (distance_matrix , 0)
print (">>>>>>>distance_matrix >>>>>>>>>" , distance_matrix)


###############################
plt.figure(figsize=(10,10))

distance_matrix = distance_matrix /np.max (distance_matrix)  # normalize
plt.imshow (distance_matrix , cmap='hot')

marks = np.arange(25)
ax = plt.gca()
plt.colorbar()
ax.set_xticks(np.arange (25))
ax.set_yticks(np.arange(25))
ax.set_xticklabels(wl_names , rotation=90)
ax.set_yticklabels( wl_names)
#plt.tight_layout()
plt.show()

plt.savefig('distance_imshow_nonlinear_5comp_300epoch_'+ str(n_val) + '.png')









