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
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

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

def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list
    
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

def Data_preprocessing(max_sentence_length, embedding_dim):
    # Word to vector
    texts = []
    with open('test_words_seg.txt', 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            line = line.split(' ')
            texts.append(line)   
    emb_model = word2vec.Word2Vec.load("word2vec.model")  
    # Convert words to index
    test_sequences = []
    for i, s in enumerate(texts):
        toks = []
        for w in s:
            if w in emb_model.wv:
                toks.append(emb_model.wv.vocab[w].index + 1) # Plus 1 to reserve index 0 for OOV words
            else:
                toks.append(0) 
        test_sequences.append(toks)
    # Pad sequence to same length
    test_x = pad_sequences(test_sequences, maxlen=max_sentence_length)     
    return test_x

def main(): 
    version = 8
    batchSize = 1024    
    max_sentence_length = 40
    embedding_dim = 128
    
    TEST_FILE = sys.argv[1]
    CHINESE_LIB = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]
    
    # Data preprocessing
    Load_text(TEST_FILE)
    Word_segmentation(CHINESE_LIB)
    test_x = Data_preprocessing(max_sentence_length, embedding_dim)

    # Infernce RNN
    model = load_model('RNN_{}.h5'.format(version))
    model.summary()
    y_pred = model.predict_classes(test_x, batch_size=batchSize)
    y_pred = flatten(y_pred)
    write_prediction_results(OUTPUT_FILE,y_pred) 
    
if __name__ == "__main__":
    main()