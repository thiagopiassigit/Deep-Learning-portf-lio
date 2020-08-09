#coding: utf-8

"""
UTILITARIOS GERAIS PARA PREPROCESSAMENTO DE DADOS
POR: Thiago Piassi Bonfogo
"""

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def result_extract(df,label):
    res = df[label]
    del df[label]
    return res

def number_class(s):
    List = list()
    for element in s:
        if(List.count(element) == 0):
            List.append(element)
    return len(List)

def number_classes(df):
    for col in df.columns:
        List = list()
        for element in df[col]:
            if(List.count(element) == 0):
                List.append(element)
        print(f'{col}: {len(List)}')
    print(f'{len(df.columns)} columns')
    
def text_to_number(s): return LabelEncoder().fit_transform(s)

def numeric_df(df):
    for col in df.columns:
        df[col] = text_to_number(df[col])
    return df

def classification_format(s): return to_categorical(s)