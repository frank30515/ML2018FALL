import sys
import numpy as np
import matplotlib.pyplot as plt
from math import log

feature_dict = {'AMP_TEMP':0, 'CH4':1, 'CO':2, 'NMHC':3, 'NO':4, 'NO2':5,
                'NOx':6, 'O3':7, 'PM10':8, 'PM2.5':9, 'RAINFALL':10, 'RH':11,
                'SO2':12, 'THC':13, 'WD_HR':14, 'WIND_DIREC':16, 'WIND_SPEED':16, 'WS_HR':17}

def load_data(file_path):
    print('Load data from {}'.format(file_path))
    data = np.genfromtxt(file_path, delimiter=',', encoding='big5')
    return data
    
def data_preprocessing(data,feature_num,hour):    
    data = data[1:,3:] # leave values

    for i in range(12): # month
        for j in range(20): # day
            if j==0:
                month_data = data[i*18*20+j*18:i*18*20+j*18+18,:]
            else:              
                month_data = np.hstack((month_data,data[i*18*20+j*18:i*18*20+j*18+18,:]))          
        if i==0:
            new_data = month_data
        else:           
            new_data = np.vstack((new_data,month_data))
    np.savetxt("new_data.csv", new_data, delimiter=",") 
    
    for i in range(12): # month   
        for j in range(20*24-9):
            for k in range(feature_num):
                if k==0:
                    feature = new_data[feature_num*i+k,j:j+9] #choose 9 hour values for one feature
                else:
                    feature = np.append(feature,new_data[feature_num*i+k,j:j+9]) 
            feature = np.append(feature,new_data[feature_num*i+9,j+9]) #label
            if i==0 and j==0:
                training_data = feature
            else:
                training_data = np.vstack((training_data,feature))                
            
    np.savetxt("training_data.csv", training_data, delimiter=",")       
    return training_data

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
    labels = data[:,-1]
    np.savetxt("features.csv", features, delimiter=",")
    np.savetxt("labels.csv", labels, delimiter=",")
    return features, labels

def linear_regression(train_data, train_labels, batch_size, epoch, learning_rate_w, learning_rate_b, regularization_term): 
    bias = 0
    weight = np.zeros(train_data.shape[1]).reshape(train_data.shape[1],1)
    
    previous_bias = 0
    previous_weight = np.zeros(train_data.shape[1]).reshape(train_data.shape[1],1)
    
    for i in range(epoch): 
        error = 0
        for j in range(int(train_data.shape[0]/batch_size)):
            # x = train_data[j*batch_size:j*batch_size+batch_size,:]
            # y_predict = bias + np.dot(x,weight)
            # y = train_labels[j*batch_size:j*batch_size+batch_size].reshape(batch_size,1)
            
            # wrong
            x = train_data[j:j+batch_size,:]
            y_predict = bias + np.dot(x,weight)
            y = train_labels[j:j+batch_size].reshape(batch_size,1)
            
            weight_gradient = (-np.dot(np.transpose(x),(y-y_predict)) +regularization_term*weight) / batch_size
            bias_gradient = -2*(np.sum((y-y_predict))) / batch_size
            
            previous_weight = previous_weight + weight_gradient**2
            previous_bias =  previous_bias + bias_gradient**2
            
            weight = weight - learning_rate_w/np.sqrt(previous_weight) * weight_gradient
            bias = bias - learning_rate_b/np.sqrt(previous_bias) * bias_gradient  
            
            error = error + (y-y_predict)**2
        
        print(np.sqrt(error/j))
        
    return weight, bias
    
def training(features, labels, batch_size, epoch, learning_rate_w, learning_rate_b, regularization_term):    
    train_error_history = []   
    val_error_history = []
    
    for i in range(int(features.shape[0]/batch_size)):
        print("-------Leaving {} Batch Out-------".format(i + 1))  
        
        # get data and label for croos validation
        train_data = features
        for j in range(batch_size):
            train_data = np.delete(train_data,i*batch_size,axis=0)            
        val_data = features[i*batch_size:i*batch_size+batch_size,:]
        
        train_labels = labels
        for j in range(batch_size):
            train_labels = np.delete(train_labels,i*batch_size)            
        val_labels = labels[i*batch_size:i*batch_size+batch_size]
 
        # training
        weight, bias = linear_regression(train_data, train_labels, batch_size, epoch, learning_rate_w, learning_rate_b, regularization_term)
        predict_labels = bias + np.dot(train_data,weight)
        train_error = np.sqrt(np.mean((train_labels-predict_labels.flatten())**2))
        # validating
        predict_labels = bias + np.dot(val_data,weight)
        val_error = np.sqrt(np.mean((val_labels-predict_labels.flatten())**2))

        print('Training Result')
        print("RMSE: {}".format(train_error))
        print('Validating Result')
        print("RMSE: {}".format(val_error))

        train_error_history.append(train_error)
        val_error_history.append(val_error)
    
    print('\n-------------------------')
    print('Average Training Result')
    print("RMSE: {}".format(np.mean(train_error_history)))
    print('Average Validating Result')
    print("RMSE: {}".format(np.mean(val_error_history)))

def main():    
    batch_size = 60
    epoch = 10000
    learning_rate_b = 0.001
    learning_rate_w = 0.001
    regularization_term = 1e-6 #2
    
    # training_data = load_data('./train.csv')
    # training_data = data_preprocessing(training_data, feature_num=18, hour=24)    
    training_data = load_data('./training_data.csv')
    features, labels = feature_extraction(training_data, feature_type=['PM2.5'])    
    
    # cross validation
    # training(features, labels, batch_size, epoch, learning_rate_w, learning_rate_b, regularization_term)
    weight, bias = linear_regression(features, labels, batch_size, epoch, learning_rate_w, learning_rate_b, regularization_term)
    # training
    predict_labels = bias + np.dot(features,weight)
    train_error = np.sqrt(np.mean((labels-predict_labels.flatten())**2))
    print('Training Result')
    print("RMSE: {}".format(train_error))
    
    np.savetxt("predict_labels.csv", predict_labels, delimiter=",")
    np.savetxt("weight.csv", weight, delimiter=",")
    np.savetxt("bias.csv", np.array([bias]), delimiter=",")
    
if __name__ == "__main__":
    main()