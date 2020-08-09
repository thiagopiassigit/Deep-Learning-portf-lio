# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:01:49 2020

@author: thiago
"""

from pandas import read_csv
from preprocessing import result_extract, numeric_df, classification_format
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

#--------------------------------------------MONTANDO CONJUNTO DE DADOS
inputs = numeric_df(read_csv('mushrooms.csv'))
result = result_extract(inputs,'class')
result = classification_format(result)
#--------------------------------------------

#--------------------------------------------SEPARANDO CONJUNTO TREINO/VALIDAÇÃO
training_input, test_input, training_result, test_result = train_test_split(
    inputs,
    result, 
    test_size=0.3
)
#--------------------------------------------

#--------------------------------------------MONTANDO REDE NEURAL

#------CONFIGURAÇÕES
units_input = int((len(inputs.columns) + 3) /2)
units = 13
monitor = EarlyStopping(patience=2)

#--------CAMADA DE ENTRADA
model = Sequential()
model.add(Dense(units_input, activation='relu',input_shape=(inputs.shape[1],)))

#---------CAMADAS OCULTAS
model.add(Dense(units, activation='relu'))
model.add(Dense(units, activation='relu'))
model.add(Dense(units, activation='relu'))

#---------CAMADA DE SAIDA
model.add(Dense(2, activation='softmax'))
#--------------------------------------------

#-------------------------------------------CONFIGURANDO TREINAMENTO
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['categorical_accuracy']
)
#-------------------------------------------

#-------------------------------------------TREINANDO REDE
model.fit(
    training_input, 
    training_result, 
    epochs=500,
    callbacks = [monitor]
)
#-------------------------------------------

#------------------------------------------AVALIANDO MODELO

#OBETENDO DADOS
score = model.evaluate(test_input,test_result)

#FORMATANDO
score = {
    'erro': str(score[0] * 100) + "%",
    'acerto': str(score[1] * 100) + "%"
}
#------------------------------------------

print(score)