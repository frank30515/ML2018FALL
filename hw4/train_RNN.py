################################################################################
#                            Machine Learning 2018                             #
#                            Hw4 : Task Description                            #
#                           Recurrent Neural Network                           #
#                         Description : training model                         #
#                       script : python3 train.py train.csv                    #
################################################################################
import numpy as np
import pandas as pd
from math import log10
import pickle
import os
import sys
import jieba
import tensorflow as tf
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau

def Load_text(text_path):
    text = open('train_text.txt', 'w', encoding='utf-8')
    with open(text_path, 'r', encoding='utf-8') as fr:
        for texts_num, line in enumerate(fr):
            line = line.strip('\n')
            line = line.replace(" ","")
            if texts_num == 0: # ignore header
                continue
            elif texts_num == 1: # prevent log0               
                text.write(line[2:]+'\n')
            elif texts_num == 120000: # prevent last newline 
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
    output = open('train_words_seg.txt', 'w', encoding='utf-8')
    with open('train_text.txt', 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            if texts_num != 119999: # prevent last newline 
                output.write('\n')
    
def Word_embedding(embedding_dim):  
    # cat train_words_seg.txt test_words_seg.txt > words_seg.txt
    sentences = word2vec.LineSentence("words_seg.txt")   
    emb_model = word2vec.Word2Vec(sentences, size=embedding_dim, window=10, min_count=5, workers=4, sg=1)
    emb_model.save("word2vec.model")
    num_words = len(emb_model.wv.vocab) + 1  # +1 for OOV words   
    # Create embedding matrix 
    embedding_matrix = np.zeros((num_words, embedding_dim), dtype=float)
    for i in range(num_words - 1):
        v = emb_model.wv[emb_model.wv.index2word[i]]
        embedding_matrix[i+1] = v   # Plus 1 to reserve index 0 for OOV words
    return embedding_matrix, num_words
    
def Text_preprocessing(text_path,chinese_lib,embedding_dim):
    Load_text(text_path)
    Word_segmentation(chinese_lib)
    embedding_matrix, num_words = Word_embedding(embedding_dim) 
    return embedding_matrix, num_words
    
def Data_preprocessing(label_path, validation_split, max_sentence_length, embedding_dim):
    texts = []
    with open('train_words_seg.txt', 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            line = line.split(' ')
            texts.append(line) 
    emb_model = word2vec.Word2Vec.load("word2vec.model")  
    # Convert words to index
    train_sequences = []
    for i, s in enumerate(texts):
        toks = []
        for w in s:
            if w in emb_model.wv:
                toks.append(emb_model.wv.vocab[w].index + 1) # Plus 1 to reserve index 0 for OOV words
            else:
                toks.append(0) 
        train_sequences.append(toks)
    # Pad sequence to same length
    data_x = pad_sequences(train_sequences, maxlen=max_sentence_length)
    # Read Label
    train_label = pd.read_csv(label_path, encoding='utf-8') 
    data_y = np.array(train_label['label'])  
    # Separate training data and validation data
    validNum = int(validation_split*data_x.shape[0])
    train_x = data_x[validNum:]
    train_y = data_y[validNum:]
    valid_x = data_x[:validNum]
    valid_y = data_y[:validNum]      
    
    return train_x, train_y, valid_x, valid_y
    
def RNN(train_x, train_y,valid_x, valid_y, batchSize, epochs, num_words, max_sentence_length, embedding_dim, embedding_matrix, verion):     
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_sentence_length,
                        trainable=False))
    model.add(GRU(512, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, input_shape=(max_sentence_length, embedding_dim)))
    model.add(GRU(512, dropout=0.4, recurrent_dropout=0.4))
    # model.add(LSTM(512, dropout=0.3, recurrent_dropout=0.5, return_sequences=True, input_shape=(max_sentence_length, embedding_dim)))
    # model.add(LSTM(512, dropout=0.3, recurrent_dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model compiling
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.summary()    
    # callbacks
    save = ModelCheckpoint('./RNN_{}.h5'.format(verion), 
                            monitor='val_acc', 
                            verbose=1, 
                            save_best_only = True) 
    earlystopping = EarlyStopping(monitor='val_acc', 
                                  patience=6, 
                                  verbose=1, 
                                  mode='max')
    # training model
    model_result = model.fit(
        train_x, 
        train_y, 
        validation_data=(valid_x, valid_y), 
        batch_size=batchSize, 
        epochs=epochs, 
        callbacks=[save])
    # evaluate
    # score = model.evaluate(train_x, train_y, batch_size=batchSize, verbose=0)
    # print ('\nTrain Acc:', score[1])
    # score = model.evaluate(valid_x, valid_y, batch_size=batchSize, verbose=0)
    # print ('\nVal Acc:', score[1])      

def DNN(train_x, train_y,valid_x, valid_y, batchSize, epochs, num_words, max_sentence_length, embedding_dim, pre_trained):     
    model = Sequential()
    model.add(Dense(512, input_shape=(max_sentence_length,embedding_dim), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model compiling
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.summary()    
    # callbacks
    save = ModelCheckpoint('./DNN_1.h5', monitor='val_acc', verbose=1, save_best_only = True) # save improved model only
    lr_reducer = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=np.sqrt(0.1), min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    # training model
    model_result = model.fit(
        train_x, 
        train_y, 
        validation_data=(valid_x, valid_y), 
        batch_size=batchSize, 
        epochs=epochs, 
        callbacks=[save])
    # evaluate
    score = model.evaluate(train_x, train_y, batch_size=batchSize, verbose=0)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(valid_x, valid_y, batch_size=batchSize, verbose=0)
    print ('\nVal Acc:', score[1])   
    
def main():
    # Allocate GPU resource
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)    
    
    TEXT_PATH = sys.argv[1]
    LABEL_PATH = sys.argv[2]
    TEST_FILE = sys.argv[3]
    CHINESE_LIB = sys.argv[4]
    
    version = 9
    validation_split = 0.2
    batchSize = 1024
    epochs = 100
    max_sentence_length = 40
    embedding_dim = 128
    # Text Preprocessing: load text, word segmentation, word embedding(pretrained)
    embedding_matrix, num_words = Text_preprocessing(TEXT_PATH,CHINESE_LIB,embedding_dim)    
    # Data Preprocessing
    train_x, train_y, valid_x, valid_y = Data_preprocessing(LABEL_PATH, validation_split, max_sentence_length, embedding_dim)    
    # Training RNN   
    RNN(train_x, train_y,valid_x, valid_y, batchSize, epochs, num_words, max_sentence_length, embedding_dim, embedding_matrix, version)
    # DNN(train_x, train_y,valid_x, valid_y, batchSize, epochs, num_words, max_sentence_length, embedding_dim, pre_trained)

    
if __name__ == '__main__':
    main()