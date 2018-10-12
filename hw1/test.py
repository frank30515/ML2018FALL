import sys
import numpy as np
import matplotlib.pyplot as plt
from math import log

feature_dict = {'AMP_TEMP':0, 'CH4':1, 'CO':2, 'NMHC':3, 'NO':4, 'NO2':5,
                'NOx':6, 'O3':7, 'PM10':8, 'PM2.5':9, 'RAINFALL':10, 'RH':11,
                'SO2':12, 'THC':13, 'WD_HR':14, 'WIND_DIREC':16, 'WIND_SPEED':16, 'WS_HR':17}

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', encoding='gb18030')
    # data = np.genfromtxt(file_path, delimiter=',')
    return data
    
def data_preprocessing(data,feature_num,hour):    
    data = data[:,2:] # leave values    
    for i in range(260):  
        for j in range(hour-9+1):
            for k in range(feature_num):
                if k==0:
                    feature = data[feature_num*i+k,j:j+9] #choose 9 hour values for one feature
                else:
                    feature = np.append(feature,data[feature_num*i+k,j:j+9]) 
                    
            if i==0 and j==0:
                testing_data = feature
            else:
                testing_data = np.vstack((testing_data,feature))   
                
    return testing_data
    

def feature_extraction(data, feature_type):
    # feature extraction
    i = 0;    
    for feature in feature_type:
        index = feature_dict[feature]
        if i == 0:
            features = data[:,index*9:index*9+9]
        else:
            features = np.hstack((features,data[:,index*9:index*9+9]))  
        i=i+1
    
    # PM2.5 ^2
    PM_square = features[:,:9]**2
    features = np.hstack((features,PM_square)) 
    
    # feature normalization BAD!!!!!!
    # features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)   
    return features
    
def write_prediction_results(file_path,features,weight,bias):
    y_predict = bias + np.dot(features,weight)
    
    output_file = []
    output_file.append(['id','value'])
    for i in range(y_predict.shape[0]):
        output_file.append(['id_'+str(i),y_predict[i]])
    
    with open(file_path, 'w') as f:
        i=0
        for item in output_file:
            i=i+1
            if i== y_predict.shape[0]+1:
                f.write(item[0])
                f.write(',')
                f.write(str(item[1]))           
            else:
                f.write(item[0])
                f.write(',')
                f.write(str(item[1]))
                f.write('\n')
            
    
def main():    
    testing_data = load_data(sys.argv[1])
    testing_data = data_preprocessing(testing_data, feature_num=18, hour=9) 
    features = feature_extraction(testing_data, feature_type=['PM2.5'])
    
    #inference
    bias = load_data('./bias.csv')
    weight = load_data('./weight.csv')
    write_prediction_results(sys.argv[2],features,weight,bias)
	
if __name__ == "__main__":
    main()