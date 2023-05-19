import re
import string
from janome.tokenizer import Tokenizer as JPTokenizer
from keras.preprocessing.text import Tokenizer
import tensorflow as tf


mispell_dict = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "'s": " is",
    "'ll": " will",
    "'d": " would",
    "'re": " are",
    "'ve": " have",
    "'m": " am",
    "wasn't": "was not",
    "didn't": "did not",
    "tryin'": "trying"
}

mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
string.punctuation += '、。【】「」『』…・〽（）〜？！｡：､；･'
CP = lambda x: x.translate(str.maketrans('', '', string.punctuation))

token_jp = JPTokenizer()
en_tokenizer = Tokenizer(filters='')
jp_tokenizer = Tokenizer(filters='')

DATA = 'data/'
TRAIN = DATA + 'train.csv'
DEV = DATA + 'dev.csv'
TEST = DATA + 'test.csv'
PICKLE = 'pickle/'
TRAINED_MODEL = 'trained_models/'

EPOCHS = 7
PATIENCE = 2

BATCH_SIZE = 64

IS_REBUILD = 1

BUFFER_SIZE = 1024
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
# val_steps_per_epoch = len(val_jp) // BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32

MAX_LENGTH = 128
