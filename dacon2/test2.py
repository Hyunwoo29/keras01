import pandas as pd
import os
import random
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
import numpy as np
path = './_data/'
def load_dataset(path, seed = 2100) : 
    data_path = os.path.join(path, 'train.csv')
    data = pd.read_csv(data_path)
    data_texts = data['사업명'] + data['과제명'] + data['요약문_연구내용'].fillna('.').astype('str')
    data_labels = data['label']
    
    random.seed(seed)
    random.shuffle(data_texts)
    random.seed(seed)
    random.shuffle(data_labels)
    
    train_texts = data_texts.loc[:data_texts.shape[0]*0.8]
    train_labels = data_labels.loc[:data_labels.shape[0]*0.8]
    
    valid_texts = data_texts.loc[data_texts.shape[0]*0.8:]
    valid_labels = data_labels.loc[data_labels.shape[0]*0.8:]
    
    return  ((data_texts, data_labels),
            (train_texts, train_labels),
            (valid_texts, valid_labels))


data_set, train_set, valid_set = load_dataset(path, seed = 2100)
test_set = pd.read_csv('./_data/test.csv')
test_set = test_set['사업명'] + test_set['과제명'] + test_set['요약문_연구내용'].fillna('.').astype('str')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

Ngram_range = (1,2)
top_k = 3000 # 최대 5만 단어 사용
token_mode = 'word' # 단어를 기준으로 tokenization 할 예정
min_document_freq = 2 # 최소 2번 이상 나타나야 함

def train_ngram_vectorize() : 
    kwargs = {
        'ngram_range' : Ngram_range,
        'dtype' : 'int32',
        'strip_accents' : False,
        'lowercase' : False,
        'decode_error' : 'replace',
        'analyzer': token_mode,
        'min_df' : min_document_freq,
        
    }
    
    vectorizer = TfidfVectorizer(**kwargs)
    
    x_train = vectorizer.fit_transform(train_set[0])
    x_val = vectorizer.transform(valid_set[0])
    
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, train_set[1].values)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    
    
    return x_train, x_val

x_train, x_val = train_ngram_vectorize()

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def single_dense(x, units):
    fc = Dense(units, activation = None)(x)
    batch = BatchNormalization()(fc)
    relu = ReLU()(batch)
    dr = Dropout(0.2)(relu)
    
    return dr

def create_model(input_shape, num_labels, learning_rate):
    x_in = Input(input_shape,)
    fc = single_dense(x_in, 256)
    fc = single_dense(fc, 128)
    fc = single_dense(fc, 64)
    
    x_out = Dense(num_labels, activation = 'softmax')(fc)
    
    model = Model(x_in, x_out)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    return model

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            ]


model = create_model(x_train.shape[1], 46, 1e-3)
history = model.fit(
                    x_train.toarray(),
                    train_set[1],
                    epochs=100,
                    callbacks=callbacks,
                    validation_data=(x_val.toarray(), valid_set[1]),
                    verbose=1,  # Logs once per epoch.
                    batch_size=314)


Ngram_range = (1,2)
top_k = 3000 # 최대 5만 단어 사용
token_mode = 'word' # 단어를 기준으로 tokenization 할 예정
min_document_freq = 2 # 최소 2번 이상 나타나야 함

def test_ngram_vectorize() : 
    kwargs = {
        'ngram_range' : Ngram_range,
        'dtype' : 'int32',
        'strip_accents' : False,
        'lowercase' : False,
        'decode_error' : 'replace',
        'analyzer': token_mode,
        'min_df' : min_document_freq,
        
    }
    
    vectorizer = TfidfVectorizer(**kwargs)
    
    x_train = vectorizer.fit_transform(data_set[0])
    x_test = vectorizer.transform(test_set)
    
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, data_set[1].values)
    x_train = selector.transform(x_train).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    
    
    return x_train, x_test

x_train, x_test = test_ngram_vectorize()

model = create_model(x_train.shape[1], 46, 1e-3)
history = model.fit(
                    x_train.toarray(),
                    data_set[1],
                    epochs=200,
                    verbose=1,  # Logs once per epoch.
                    batch_size=312)

prediction = model.predict(x_test.toarray())

sample = pd.read_csv('./_data/sample_submission.csv')
sample['label'] = np.argmax(prediction, axis = 1)
sample.head()

sample['label'].nunique()
sample.to_csv('./_data/bert_5.csv', index = False)