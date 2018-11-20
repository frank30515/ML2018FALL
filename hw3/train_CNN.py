################################################################################
#                            Machine Learning 2018                             #
#                     Hw3 : Image Sentiment Classification                     #
#                         Convolutional Neural Network                         #
#                         Description : training model                         #
#                       script : python3 train.py train.csv                    #
################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rand
import os
import sys
import tensorflow as tf
from scipy import misc
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau

def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
    
def load_data(path):
    # load data from path
    train_data = pd.read_csv(path)    
    train_x = train_data['feature']
    train_y = np.array(train_data['label']) 
    x = []
    for i in range(train_x.shape[0]):
        x.append(train_x[i].split(' '))
    train_x = np.array(x, dtype=float)
    train_x = train_x/255          
    return train_x, train_y        
    
def data_preprocessing(data_x, data_y, validNum, classNum):
    train_x = data_x[validNum:]
    train_y = data_y[validNum:]
    valid_x = data_x[:validNum]
    valid_y = data_y[:validNum]
    # reshape feature to fit the model 48*48
    train_x = train_x.reshape(train_x.shape[0],48,48,1) 
    valid_x = valid_x.reshape(valid_x.shape[0],48,48,1)
    # reshape lable into one hot encoding
    train_y = to_categorical(train_y, classNum)    
    valid_y = to_categorical(valid_y, classNum)    
    return train_x, train_y, valid_x, valid_y

def CNN(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs):                               
    input_shape = (train_x.shape[1], train_x.shape[2], 1)
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same'),
    BatchNormalization(),    
    LeakyReLU(alpha=0.1),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),   
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Dropout(0.3),
    
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(), 
    LeakyReLU(alpha=0.1),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),  
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Dropout(0.3),
    
    Conv2D(192, (3, 3), padding='same',),
    BatchNormalization(), 
    LeakyReLU(alpha=0.1),
    Conv2D(192, (3, 3), padding='same',),
    BatchNormalization(), 
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Dropout(0.3),
    
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(), 
    LeakyReLU(alpha=0.1),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(), 
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Dropout(0.3),
    
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(), 
    LeakyReLU(alpha=0.1),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	Dropout(0.3),
    
    Flatten(),
    Dense(512,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(classNum, activation='softmax')
    ])
    # model compiling
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # data augmentation
    data_gen = ImageDataGenerator(rotation_range=10.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)        
    data_gen.fit(train_x)  
    # callbacks
    save = ModelCheckpoint('./CNN_16.h5', monitor='val_acc', verbose=1, save_best_only = True) # save improved model only
    lr_reducer = ReduceLROnPlateau( monitor='val_acc', patience=5, verbose=1, factor=np.sqrt(0.1), min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./CNN_16.csv')
    # training model
    model_result = model.fit_generator(
        data_gen.flow(train_x, train_y, batchSize), 
        validation_data=(valid_x, valid_y),
        steps_per_epoch=train_x.shape[0]//batchSize, 
        epochs=epochs,
        callbacks=[save])

    score = model.evaluate(train_x,train_y)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(valid_x,valid_y)
    print ('\nVal Acc:', score[1])    
    
    show_train_history(model_result)


def VGG_16(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs, show):                     
    input_shape = (train_x.shape[1], train_x.shape[2], 1)
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(classNum, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # data augmentation
    data_gen = ImageDataGenerator(rotation_range=10.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)        
    data_gen.fit(train_x)  
    # callbacks
    save = ModelCheckpoint('./CNN_9.h5', monitor='val_acc', verbose=1, save_best_only = True) # save improved model only
    lr_reducer = ReduceLROnPlateau( monitor='val_acc', patience=5, verbose=1, factor=np.sqrt(0.1), min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./CNN_9.csv')
    # training model
    model_result = model.fit_generator(
        data_gen.flow(train_x, train_y, batchSize), 
        validation_data=(valid_x, valid_y),
        steps_per_epoch=train_x.shape[0]//batchSize, 
        epochs=epochs,
        callbacks=[save])

    score = model.evaluate(train_x,train_y)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(valid_x,valid_y)
    print ('\nVal Acc:', score[1])    
    
    if show == True:
        show_train_history(model_result)
    
def VGG_19(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs, show):                     
    input_shape = (train_x.shape[1], train_x.shape[2], 1)
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),    
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(classNum, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # data augmentation
    data_gen = ImageDataGenerator(rotation_range=10.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)        
    data_gen.fit(train_x)  
    # callbacks
    save = ModelCheckpoint('./CNN_11.h5', monitor='val_acc', verbose=1, save_best_only = True) # save improved model only
    lr_reducer = ReduceLROnPlateau( monitor='val_acc', patience=5, verbose=1, factor=np.sqrt(0.1), min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./CNN_11.csv')
    # training model
    model_result = model.fit_generator(
        data_gen.flow(train_x, train_y, batchSize), 
        validation_data=(valid_x, valid_y),
        steps_per_epoch=train_x.shape[0]//batchSize, 
        epochs=epochs,
        callbacks=[save])

    score = model.evaluate(train_x,train_y)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(valid_x,valid_y)
    print ('\nVal Acc:', score[1])    
    
    if show == True:
        show_train_history(model_result)

    
def VGG_11(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs, show):                     
    input_shape = (train_x.shape[1], train_x.shape[2], 1)
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same'),
    BatchNormalization(),     
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),  
    Activation('relu'),   
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),  
    Activation('relu'),   
    Conv2D(256, (3, 3), padding='same',),
    BatchNormalization(),  
    Activation('relu'),   
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),  
    Activation('relu'),   
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),  
    Activation('relu'),   
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding='same',),  
    BatchNormalization(),  
    Activation('relu'),   
    Conv2D(512, (3, 3), padding='same',),
    BatchNormalization(),  
    Activation('relu'),   
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(classNum, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # data augmentation
    data_gen = ImageDataGenerator(rotation_range=10.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)        
    data_gen.fit(train_x)  
    # callbacks
    save = ModelCheckpoint('./CNN_8.h5', monitor='val_acc', verbose=1, save_best_only = True) # save improved model only
    lr_reducer = ReduceLROnPlateau( monitor='val_acc', patience=5, verbose=1, factor=np.sqrt(0.1), min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./CNN_8.csv')
    # training model
    model_result = model.fit_generator(
        data_gen.flow(train_x, train_y, batchSize), 
        validation_data=(valid_x, valid_y),
        steps_per_epoch=train_x.shape[0]//batchSize, 
        epochs=epochs,
        callbacks=[save])

    score = model.evaluate(train_x,train_y)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(valid_x,valid_y)
    print ('\nVal Acc:', score[1])    
    
    if show == True:
        show_train_history(model_result)
 
def main():    
    classNum = 7
    validNum = 5000
    batchSize = 128
    epochs = 600
    # data preprocessing
    train_x, train_y = load_data(sys.argv[1])
    train_x, train_y, valid_x, valid_y = data_preprocessing(train_x, train_y, validNum, classNum)
    # training CNN
    CNN(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs)
    # VGG_11(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs, show=False)
    # VGG_16(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs, show=False)
    # VGG_19(train_x, train_y, valid_x, valid_y, classNum, batchSize, epochs, show=False)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    main()