import numpy as np

import constants
from constants import *
from tqdm.notebook import tqdm
from keras.utils import pad_sequences
import pickle
import tensorflow as tf


def preprocess_en(text) -> str:
    def replace(match):
        return mispell_dict[match.group(0)]

    text = mispell_re.sub(replace, text)
    text = '[START] ' + text + ' [END]'
    return CP(text)


def preprocess_jp(jp_text) -> str:
    jp_text = CP(' '.join([word for word in token_jp.tokenize(jp_text, wakati=True) if word != ' ']))
    return_text = '[START] ' + jp_text + ' [END]'
    return return_text


def process_data(data_file, is_train=False):
    all_en, all_jp = [], []

    with open(data_file, encoding='utf-8') as f:
        lines = f.readlines()
        f.close()

    for line in tqdm(lines, desc='Processing dataset: {}'.format(
            'train' if is_train else 'dev' if data_file == DEV else 'test')):
        segments = line.split(',')
        if len(segments) == 2:
            en_line = segments[0].strip()
            jp_line = segments[1].strip()
        else:
            _ = segments[0]
            jp_line = segments[-1].strip()
            en_line = ' '.join(segments[1:len(segments) - 1]).strip()

        print(en_line)
        print(jp_line)
        all_en.append(en_line)
        all_jp.append(jp_line)

    return all_en, all_jp


def data_builder(outfile_data):
    train_en, train_jp = process_data(TRAIN)
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


def get_processors(data):
    en_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_en,
        ragged=True)
    en_processor.adapt(data.map(lambda context, target: context))
    jp_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_jp,
        ragged=True)
    jp_processor.adapt(data.map(lambda context, target: target))

    def process_text(context, target):
        context = en_processor(context).to_tensor()
        target = jp_processor(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out

    return en_processor, jp_processor, process_text
