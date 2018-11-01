import sys
import numpy as np
import pandas as pd
import model

feature_dict = {'LIMIT_BAL':[0], 'SEX':[1], 'EDUCATION':[2], 'MARRIAGE':[3], 'AGE':[4], 'PAY_0':[5],
                'PAY_2':[6], 'PAY_3':[7], 'PAY_4':[8], 'PAY_5':[9], 'PAY_6':[10], 'BILL_AMT1':[11],
                'BILL_AMT2':[12], 'BILL_AMT3':[13], 'BILL_AMT4':[14], 'BILL_AMT5':[16], 'BILL_AMT6':[16], 'PAY_AMT1':[17]
                , 'PAY_AMT2':[18], 'PAY_AMT3':[19], 'PAY_AMT4':[20], 'PAY_AMT5':[21], 'PAY_AMT6':[22], 'SEX_hot':range(23,25)
                , 'EDUCATION_hot':range(25,32), 'MARRIAGE_hot':range(32,36), 'PAY_0_hot':range(36,47), 'PAY_2_hot':range(47,58)
                , 'PAY_3_hot':range(58,69), 'PAY_4_hot':range(69,80), 'PAY_5_hot':range(80,91), 'PAY_6_hot':range(91,102)}
    
def feature_extraction(data, feature_type):
    features = []
    for feature in feature_type:
        index = feature_dict[feature]
        for i in index:
            features.append(data[:,i])
    features = np.array(features).T    
    return features

def write_prediction_results(file_path,y_predict):    
    output_file = []
    output_file.append(['id','value'])
    for i in range(len(y_predict)):
        output_file.append(['id_'+str(i),y_predict[i]])
    
    with open(file_path, 'w') as f:
        i=0
        for item in output_file:
            i=i+1
            if i== len(y_predict)+1:
                f.write(item[0])
                f.write(',')
                f.write(str(item[1]))           
            else:
                f.write(item[0])
                f.write(',')
                f.write(str(item[1]))
                f.write('\n')
        
def one_hot_encoding(file_path):
    df = pd.read_csv(file_path,encoding='big5') 
    df2 = pd.get_dummies(df['SEX'])
    df3 = pd.get_dummies(df['EDUCATION'])
    df4 = pd.get_dummies(df['MARRIAGE'])
    df5 = pd.get_dummies(df['PAY_0'])
    df6 = pd.get_dummies(df['PAY_2'])
    df7 = pd.get_dummies(df['PAY_3'])
    df8 = pd.get_dummies(df['PAY_4'])
    df9 = pd.get_dummies(df['PAY_5'])
    df10 = pd.get_dummies(df['PAY_6'])    
    # fill 0 if PAY doesn't have some values
    for i in range(-2,9):        
        if i not in list(df5.columns):
            df5.insert(i+2, i, 0)
        if i not in list(df6.columns):
            df6.insert(i+2, i, 0)
        if i not in list(df7.columns):
            df7.insert(i+2, i, 0)
        if i not in list(df8.columns):
            df8.insert(i+2, i, 0)
        if i not in list(df9.columns):
            df9.insert(i+2, i, 0)
        if i not in list(df10.columns):
            df10.insert(i+2, i, 0)
            
    frames = [df,df2,df3,df4,df5,df6,df7,df8,df9,df10]
    frames = pd.concat(frames,axis=1)
    feature = frames.values
    return feature
    
def main():   
    train_mean = np.genfromtxt("train_mean_best.csv", delimiter=',')
    train_std = np.genfromtxt("train_std_best.csv", delimiter=',')
    bias = np.genfromtxt("bias_best.csv", delimiter=',')
    weight = np.genfromtxt("weight_best.csv", delimiter=',')
    
    test_x = one_hot_encoding(sys.argv[3]) 
    test_x[:,:24] = (test_x[:,:24]-train_mean)/train_std
    test_x = feature_extraction(test_x,  \
    ['PAY_0_hot','PAY_2_hot','PAY_3_hot','PAY_4_hot','PAY_5_hot','PAY_6_hot']) 
   
    y_pred = []        
    z = np.dot(test_x, weight) + bias
    for i in range(z.shape[0]):
        if z[i] > 0 :
            y_pred.append(1)
        else:
            y_pred.append(0)
            
    write_prediction_results(sys.argv[4],y_pred)    
    
if __name__ == "__main__":
    main()