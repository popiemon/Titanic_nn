import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn import preprocessing
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_csv = 'train.csv'
    train_df = pd.read_csv(train_csv, header = 0)
    
    test_csv = 'test.csv'
    test_df = pd.read_csv(test_csv, header = 0)
    test_df_origin = test_df
 
    #Sex. male=0, female=1
    train_df = train_df.replace({'Sex':{'male':0, 'female':1}})
    test_df = test_df.replace({'Sex':{'male':0, 'female':1}})
    
    #Embarked. C=0, Q=1, S=2
    train_df = train_df.replace({'Embarked':{'C':0, 'Q':1, 'S':2}})
    test_df = test_df.replace({'Embarked':{'C':0, 'Q':1, 'S':2}})
    
    #drop. Name, Ticket, Cabin
    train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
    test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

    train_df = train_df.dropna(how='any') #drop rows contain NaN

    #x_column = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_column = ['Sex', 'Age', 'SibSp', 'Parch']
    y_column = ['Survived']

    x_train = train_df[x_column]
    y_train = train_df[y_column]
    
    mxscaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1.1)) #x normalization 
    mxscaler.fit(x_train)
    mx_train = mxscaler.transform(x_train)
    
    myscaler = preprocessing.MinMaxScaler(feature_range=(-0.5, 0.5)) #y normalization
    myscaler.fit(y_train)
    my_train = myscaler.transform(y_train)
    
    Titanicmodel = Sequential()
    Titanicmodel.add(Dense(activation='tanh', input_dim=len(x_column), units=5))
    Titanicmodel.add(Dense(units=6, activation='elu'))
    Titanicmodel.add(Dense(units=6, activation='elu'))
    Titanicmodel.add(Dense(units=6, activation='elu'))    
    Titanicmodel.add(Dense(units=1, activation='tanh'))
    Titanicmodel.compile(loss='mean_squared_error', optimizer=Adam(learning_rate = 0.001), metrics=['accuracy'])
    
    history = Titanicmodel.fit(mx_train, my_train, batch_size = 64, epochs = 3000)
    
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('mean_squared_error')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig('loss.png')
    plt.close()
    
    x_test = test_df[x_column]
    mx_test = mxscaler.transform(x_test)
    predict_y = Titanicmodel.predict(mx_test)
    
    y_test = myscaler.inverse_transform(predict_y)
    y_test = np.where(y_test < 0.5, 0, 1)
    
    test_df_origin['Survived'] = y_test
    
    y_sub_column = ['PassengerId', 'Survived']
    sub_df = test_df_origin[y_sub_column]
    sub_df.to_csv('Submission.csv', index=False)
    

