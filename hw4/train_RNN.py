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
    sentences = word2vec.LineSentence("train_words_seg.txt")
    model = word2vec.Word2Vec(sentences, size=embedding_dim)
    model.save("word2vec.model")
    
def Text_preprocessing(text_path,chinese_lib,embedding_dim):
    Load_text(text_path)
    Word_segmentation(chinese_lib)
    Word_embedding(embedding_dim) 

def Data_preprocessing(label_path, validation_split, max_sentence_length, embedding_dim):
    # Word to vector
    texts = []
    with open('train_words_seg.txt', 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            texts.append(line)    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    encoded_docs = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index # how many uniques word in your context
    vocab_size = len(word_index) + 1 
    with open('tokenizer.pkl', 'wb') as tok_file:
        pickle.dump(tokenizer, tok_file)
    data_x = pad_sequences(encoded_docs, maxlen=max_sentence_length, padding='post')
    # Read Label
    train_label = pd.read_csv(label_path, encoding='utf-8') 
    data_y = np.array(train_label['label'])    
    # Separate training data and validation data
    validNum = int(validation_split*data_x.shape[0])
    train_x = data_x[validNum:]
    train_y = data_y[validNum:]
    valid_x = data_x[:validNum]
    valid_y = data_y[:validNum]    
    
    # Load the whole embedding into memory
    # embeddings_index = dict()
    # f = open( '../glove_data/glove.6B/glove.6B.100d.txt' )
    # for line in f:
        # values = line.split()
        # word = values[ 0 ]
        # coefs = asarray(values[ 1 :], dtype= 'float32' )
        # embeddings_index[word] = coefs
    # f.close()
    
    # Create a weight matrix for words in training docs
    embedding_matrix = []
    # embedding_matrix = zeros((vocab_size, 100 ))
    # for word, i in t.word_index.items():
        # embedding_vector = embeddings_index.get(word)
        # if embedding_vector is  not  None :
            # embedding_matrix[i] = embedding_vector

    
    return train_x, train_y, valid_x, valid_y, vocab_size, embedding_matrix
    
def RNN(train_x, train_y,valid_x, valid_y, batchSize, epochs, vocab_size, max_sentence_length, embedding_dim, pre_trained):     
    model = Sequential()
    if pre_trained == False:
        model.add(Embedding(vocab_size, output_dim=embedding_dim, input_length=max_sentence_length, trainable=True, mask_zero=False))
    # model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # model compiling
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.summary()    
    # callbacks
    save = ModelCheckpoint('./RNN_1.h5', monitor='val_acc', verbose=1, save_best_only = True) # save improved model only
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
    
    validation_split = 0.2
    batchSize = 256
    epochs = 10
    max_sentence_length = 50
    embedding_dim = 3
    pre_trained = False    
    # Text Preprocessing: load text, word segmentation, word embedding(pretrained)
    # Text_preprocessing(TEXT_PATH,CHINESE_LIB,embedding_dim)    
    # Data Preprocessing
    train_x, train_y, valid_x, valid_y, vocab_size, embedding_matrix = Data_preprocessing(LABEL_PATH, validation_split, max_sentence_length, embedding_dim)    
    # Training RNN   
    RNN(train_x, train_y,valid_x, valid_y, batchSize, epochs, vocab_size, max_sentence_length, embedding_dim, pre_trained)
    # Write paramter
    paramter = [max_sentence_length]
    np.savetxt('paramter.txt', paramter)
    
if __name__ == '__main__':
    main()