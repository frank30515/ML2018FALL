################################################################################
#                            Machine Learning 2018                             #
#                           Hw4 : Task Description                             #
#                          Recurrent Neural Network                            #
#                   description : use model to get prediction                  #
#                   script : python3 test.py test.csv ans.csv                  #
################################################################################
import numpy as np
import pandas as pd
from math import log10
import pickle
import os
import sys
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

def write_prediction_results(file_path,y_predict):    
    output_file = []
    output_file.append(['id','label'])
    for i in range(len(y_predict)):
        output_file.append([str(i),y_predict[i][0]])
    
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

def Load_text(text_path):
    text = open('test_text.txt', 'w', encoding='utf-8')
    with open(text_path, 'r', encoding='utf-8') as fr:
        for texts_num, line in enumerate(fr):
            line = line.strip('\n')
            line = line.replace(" ","")
            if texts_num == 0: # ignore header
                continue
            elif texts_num == 1: # prevent log0               
                text.write(line[2:]+'\n')
            elif texts_num == 80000: # prevent last newline 
                idx = int(log10(texts_num-1))+2
                text.write(line[idx:]) 
            else:
                idx = int(log10(texts_num-1))+2
                text.write(line[idx:]+'\n')   
                
def Word_segmentation(chinese_lib):
    # jieba custom setting.
    jieba.set_dictionary(chinese_lib)  
    
    # load stopwords set
    stopword_set=[]
    with open('stopWords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.append(stopword.strip('\n'))
    
    # word segmentation    
    output = open('test_words_seg.txt', 'w', encoding='utf-8')
    with open('test_text.txt', 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            if texts_num != 79999: # prevent last newline 
                output.write('\n')

                
def Data_preprocessing(max_sentence_length):
    # Word to vector
    texts = []
    with open('test_words_seg.txt', 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            texts.append(line)    
    tokenizer = pickle.load(open('tokenizer.pkl','rb'))
    encoded_docs = tokenizer.texts_to_sequences(texts)
    test_x = pad_sequences(encoded_docs, maxlen=max_sentence_length, padding='post')    
    return test_x

def main(): 
    paramter = [np.genfromtxt('paramter.txt')]
    batchSize = 256    
    max_sentence_length = int(paramter[0])    
    
    TEST_FILE = sys.argv[1]
    CHINESE_LIB = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]
    
    # Data preprocessing
    # Load_text(TEST_FILE)
    # Word_segmentation(CHINESE_LIB)
    test_x = Data_preprocessing(max_sentence_length)

    # Infernce RNN
    model = load_model('RNN_1.h5')
    model.summary()
    y_pred = model.predict_classes(test_x, batch_size=batchSize)
    write_prediction_results(OUTPUT_FILE,y_pred)    
    
if __name__ == "__main__":
    main()