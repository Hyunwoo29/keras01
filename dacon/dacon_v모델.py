import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm


import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score,f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import *

train=pd.read_csv('./_data/train.csv')
test=pd.read_csv('./_data/test.csv')
sample_submission=pd.read_csv('./_data/sample_submission.csv')

os.environ["CUDA_VISIBLE_DEVICES"]="0"

train=train[['과제명', '요약문_연구내용','label']]
test=test[['과제명', '요약문_연구내용']]
train['요약문_연구내용'].fillna('NAN', inplace=True)
test['요약문_연구내용'].fillna('NAN', inplace=True)

train['data']=train['과제명']+train['요약문_연구내용']
test['data']=test['과제명']+test['요약문_연구내용']

tf.random.set_seed(2000)
np.random.seed(2000)
BATCH_SIZE = 64
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
MAX_LEN=44

from transformers import *
tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased',  cache_dir='bert_ckpt', do_lower_case=False)

def bert_tokenizer(sent, MAX_LEN):
    
    encoded_dict=tokenizer.encode_plus(
    text = sent, 
    add_special_tokens=True, 
    max_length=MAX_LEN, 
    pad_to_max_length=True, 
    return_attention_mask=True,
    truncation = True)
    
    input_id=encoded_dict['input_ids']
    attention_mask=encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']
    
    return input_id, attention_mask, token_type_id

input_ids =[]
attention_masks =[]
token_type_ids =[]
train_data_labels = []

def clean_text(sent):
    sent_clean=re.sub("[^가-힣ㄱ-하-ㅣ]", " ", sent)
    return sent_clean

for train_sent, train_label in zip(train['data'], train['label']):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(clean_text(train_sent), MAX_LEN=MAX_LEN)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        #########################################
        train_data_labels.append(train_label)
        
    except Exception as e:
        print(e)
        print(train_sent)
        pass

train_input_ids=np.array(input_ids, dtype=int)
train_attention_masks=np.array(attention_masks, dtype=int)
train_token_type_ids=np.array(token_type_ids, dtype=int)
###########################################################
train_inputs=(train_input_ids, train_attention_masks, train_token_type_ids)
train_labels=np.asarray(train_data_labels, dtype=np.int32)


class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()

        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), 
                                                name="classifier")
        
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        
        #outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits

cls_model = TFBertClassifier(model_name='bert-base-multilingual-cased',
                                  dir_path='bert_ckpt',
                                  num_class=46)

# 학습 준비하기
optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model_name = "tf2_bert_classifier"

# overfitting을 막기 위한 ealrystop 추가
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=4)
# min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
# patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)\

checkpoint_path = os.path.join(model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

# 학습과 eval 시작
history = cls_model.fit(train_inputs, train_labels, epochs=10, batch_size=284,
                    validation_split = VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

input_ids =[]
attention_masks =[]
token_type_ids =[]
train_data_labels = []

def clean_text(sent):
    sent_clean=re.sub("[^가-힣ㄱ-하-ㅣ]", " ", sent)
    return sent_clean

for test_sent in test['data']:
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(clean_text(test_sent), MAX_LEN=40)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        #########################################
       
    except Exception as e:
        print(e)
        print(test_sent)
        pass
    
test_input_ids=np.array(input_ids, dtype=int)
test_attention_masks=np.array(attention_masks, dtype=int)
test_token_type_ids=np.array(token_type_ids, dtype=int)
###########################################################
test_inputs=(test_input_ids, test_attention_masks, test_token_type_ids)

results = cls_model.predict(test_inputs)
results=tf.argmax(results, axis=1)
sample_submission['label']=results
sample_submission.to_csv('./_data/bert_baseline1.csv', index=False)