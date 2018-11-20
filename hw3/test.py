################################################################################
#                            Machine Learning 2018                             #
#                      Hw3 : Image Sentiment Classification                    #
#                         Convolutional Neural Network                         #
#                   description : use model to get prediction                  #
#                   script : python3 test.py test.csv ans.csv                  #
################################################################################
import pandas as pd
import numpy as np
import sys
from scipy import misc
from keras.models import load_model

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
    
def load_data(path):
    # load data from path
    test_data = pd.read_csv(path)    
    test_x = test_data['feature']
    x = []
    for i in range(test_x.shape[0]):
        x.append(test_x[i].split(' '))
    test_x = np.array(x, dtype=float)
    test_x = test_x/255    
    return test_x    

def data_preprocessing(data_x):
    # reshape feature to fit the model 48*48
    test_x = data_x.reshape(data_x.shape[0],48,48,1)   
    return test_x      
  
  
def write_prediction_results(file_path,y_predict):    
    output_file = []
    output_file.append(['id','label'])
    for i in range(len(y_predict)):
        output_file.append([str(i),y_predict[i]])
    
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

def main():       
    classNum = 7
    # data preprocessing
    test_x = load_data(sys.argv[1])
    test_x = data_preprocessing(test_x)

    # infernce CNN
    model = load_model('CNN_14.h5')
    model.summary()
    y_pred = model.predict_classes(test_x)            
    write_prediction_results(sys.argv[2],y_pred)    
    
if __name__ == "__main__":
    main()