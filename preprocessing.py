import numpy as np

import constants
from constants import *
from tqdm import tqdm
from keras.utils import pad_sequences
import pickle
import tensorflow as tf
import tensorflow_text as tf_text


def preprocess_en(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    for k in mispell_dict:
        text = tf.strings.regex_replace(text, k, mispell_dict[k])
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    # text = mispell_re.sub(replace, text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def preprocess_jp(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.regex_replace(text, '、。【】「」『』…・〽（）〜？！｡：､；･', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def process_data(data_file, is_train=False, size='full'):
    all_en, all_jp = [], []

    with open(data_file, encoding='utf-8') as f:
        lines = f.readlines()
        f.close()

    if size == 'full':
        num_sample = int(len(lines))
    elif size == 'medium':
        num_sample = int(len(lines) // 100 * 50)
    elif size == 'base':
        num_sample = int(len(lines) // 100 * 10)
    elif size == 'small':
        num_sample = int(len(lines) // 100 * 5)
    else:
        num_sample = int(len(lines) // 100)

    for line in tqdm(lines[:num_sample], desc='Processing dataset: {}'.format(
            'train' if is_train else 'dev' if data_file == DEV else 'test')):
        segments = line.split(',')
        if len(segments) == 2:
            en_line = segments[0].strip()
            jp_line = segments[1].strip()
        else:
            _ = segments[0]
            jp_line = segments[-1].strip()
            jp_line = CP(' '.join([word for word in token_jp.tokenize(jp_line, wakati=True) if word != ' ']))
            en_line = ' '.join(segments[1:len(segments) - 1]).strip()

        all_en.append(en_line)
        all_jp.append(jp_line)

    return all_en, all_jp


def data_builder(outfile_data):
    train_en, train_jp = process_data(TRAIN, is_train=True, size='tiny')
    dev_en, dev_jp = process_data(DEV)
    test_en, test_jp = process_data(TEST)

    data_dict = {
        'en': [train_en, dev_en, test_en],
        'jp': [train_jp, dev_jp, test_jp]
    }

    with open(outfile_data, 'wb') as f:
        pickle.dump(data_dict, f)
        f.close()

    dataset_train = (tf.data.Dataset.from_tensor_slices((train_en, train_jp))
                     .shuffle(BUFFER_SIZE)
                     .batch(BATCH_SIZE, drop_remainder=True))
    dataset_dev = (tf.data.Dataset.from_tensor_slices((dev_en, dev_jp)).batch(BATCH_SIZE, drop_remainder=True))

    dataset_test = (tf.data.Dataset.from_tensor_slices((test_en, test_jp)).batch(BATCH_SIZE, drop_remainder=True))

    return dataset_train, dataset_dev, dataset_test


def data_loader(infile_data):
    print('Loading data from datasets...')
    with open(infile_data, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()

    train_en, dev_en, test_en = data_dict['en']
    train_jp, dev_jp, test_jp = data_dict['jp']

    dataset_train = (tf.data.Dataset.from_tensor_slices((train_en, train_jp))
                     .shuffle(BUFFER_SIZE)
                     .batch(BATCH_SIZE, drop_remainder=True))
    dataset_dev = (tf.data.Dataset.from_tensor_slices((dev_en, dev_jp)).batch(BATCH_SIZE, drop_remainder=True))

    dataset_test = (tf.data.Dataset.from_tensor_slices((test_en, test_jp)).batch(BATCH_SIZE, drop_remainder=True))

    return dataset_train, dataset_dev, dataset_test


def get_processors(data_train, data_val, data_test):
    en_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_en,
        ragged=True)
    en_processor.adapt(data_train.map(lambda context, target: context))
    jp_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_jp,
        ragged=True)
    jp_processor.adapt(data_train.map(lambda context, target: target))

    def process_text(context, target):
        context = en_processor(context).to_tensor()
        target = jp_processor(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out

    data_train = data_train.map(process_text, tf.data.AUTOTUNE)
    data_val = data_val.map(process_text, tf.data.AUTOTUNE)
    data_test = data_test.map(process_text, tf.data.AUTOTUNE)

    return en_processor, jp_processor, data_train, data_val, data_test
