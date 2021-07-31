from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.datasets import reuters
from icecream import ic
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# ic(train_data.shape, train_labels.shape)
# ic(test_data.shape, test_labels.shape)
# ic| train_data.shape: (8982,), train_labels.shape: (8982,)
# ic| test_data.shape: (2246,), test_labels.shape: (2246,)

word_index = reuters.get_word_index()
# ic(word_index) 'inspecting': 25622,
#                  'inspection': 3469,
#                  'inspections': 13521 
# 키와 벨류 딕셔너리 형태 inspecting은 25622번째 인덱스이다.

# ic(len(word_index))
#ic| len(word_index): 30979

#단어를 텍스트로 디코딩
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
decoded_news = ' '.join([reverse_word_index.get(i - 3, '?' ) for i in train_data[0]])
# ic(decoded_news)
# ic| decoded_news: ('? ? ? said as a result of its december acquisition of space co it 
# expects '
#                    'earnings per share in 1987 of 1 15 to 1 30 dlrs per share up from 
# 70 cts in '
#                    '1986 the company said pretax net should rise to nine to 10 mln dlrs from six '
#                    'mln dlrs in 1986 and rental operation revenues to 19 to 22 mln dlrs from 12 '
#                    '5 mln dlrs it said cash flow per share this year should be 2 50 to three '
#                    'dlrs reuter 3')

#  데이터를 벡터로 변환
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_lables = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

ic(one_hot_train_lables)