import constants
from constants import *
from tqdm import tqdm
from keras.utils import pad_sequences
import pickle


def preprocess_en(text) -> str:
    def replace(match):
        return mispell_dict[match.group(0)]

    text = mispell_re.sub(replace, text)
    return CP(text)


def preprocess_jp(jp_text) -> str:
    return CP(' '.join([word for word in token_jp.tokenize(jp_text, wakati=True) if word != ' ']))


def sequences(texts, tokenizer):
    res = []
    for text in texts:
        seq = []
        for w in text.split():
            try:
                seq.append(tokenizer.word_index[w])
            except Exception as e:
                seq.append(tokenizer.word_index['unk'])
        res.append(seq)
    return res


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

        en_line = 'bos ' + preprocess_en(en_line) + ' eos'
        jp_line = 'bos ' + preprocess_jp(jp_line) + ' eos'
        all_en.append(en_line)
        all_jp.append(jp_line)

    if is_train:
        all_en.append('unk unk unk')
        all_jp.append('unk unk unk')

    return all_en, all_jp


def get_tokenized_data(processed_data, tokenizer, is_train=False):
    if is_train:
        return tokenizer.texts_to_sequences(processed_data)
    return sequences(processed_data, tokenizer)


def data_builder(outfile_data, outfile_token):
    train_en, train_jp = process_data(TRAIN, is_train=True)
    dev_en, dev_jp = process_data(DEV)
    test_en, test_jp = process_data(TEST)

    en_tokenizer.fit_on_texts(train_en)
    jp_tokenizer.fit_on_texts(train_jp)

    train_token_en = get_tokenized_data(train_en, en_tokenizer, is_train=True)
    train_token_jp = get_tokenized_data(train_jp, jp_tokenizer, is_train=True)

    dev_token_en = get_tokenized_data(dev_en, en_tokenizer)
    dev_token_jp = get_tokenized_data(dev_jp, jp_tokenizer)

    test_token_en = get_tokenized_data(test_en, en_tokenizer)
    test_token_jp = get_tokenized_data(test_jp, jp_tokenizer)

    train_token_en = pad_sequences(train_token_en, padding='post', maxlen=constants.MAX_LENGTH)
    train_token_jp = pad_sequences(train_token_jp, padding='post', maxlen=constants.MAX_LENGTH)

    dev_token_en = pad_sequences(dev_token_en, padding='post', maxlen=constants.MAX_LENGTH)
    dev_token_jp = pad_sequences(dev_token_jp, padding='post', maxlen=constants.MAX_LENGTH)

    test_token_en = pad_sequences(test_token_en, padding='post', maxlen=constants.MAX_LENGTH)
    test_token_jp = pad_sequences(test_token_jp, padding='post', maxlen=constants.MAX_LENGTH)

    data_dict = {
        'en': [train_token_en, dev_token_en, test_token_en],
        'jp': [train_token_jp, dev_token_jp, test_token_jp]
    }

    with open(outfile_data, 'wb') as f:
        pickle.dump(data_dict, f)
        f.close()

    tokenizer_dict = {
        'en': en_tokenizer,
        'jp': jp_tokenizer
    }

    with open(outfile_token, 'wb') as f:
        pickle.dump(tokenizer_dict, f)
        f.close()

    dataset_train = (tf.data.Dataset.from_tensor_slices({'en': train_token_en, 'jp': train_token_jp})
                     .shuffle(BUFFER_SIZE)
                     .batch(BATCH_SIZE, drop_remainder=True))
    dataset_dev = (tf.data.Dataset.from_tensor_slices({'en': dev_token_en,
                                                       'jp': dev_token_jp}).batch(BATCH_SIZE, drop_remainder=True))

    dataset_test = (tf.data.Dataset.from_tensor_slices({'en': test_token_jp,
                                                        'jp': test_token_jp}).batch(BATCH_SIZE, drop_remainder=True))

    return dataset_train, dataset_dev, dataset_test


def data_loader(infile_data, infile_token):
    print('Loading data from datasets...')
    with open(infile_data, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()
    with open(infile_token, 'rb') as f:
        token_dict = pickle.load(f)

    en_token = token_dict['en']
    jp_token = token_dict['jp']

    train_token_en, dev_token_en, test_token_en = data_dict['en']
    train_token_jp, dev_token_jp, test_token_jp = data_dict['jp']

    dataset_train = (tf.data.Dataset.from_tensor_slices({'en': train_token_en, 'jp': train_token_jp})
                     .shuffle(BUFFER_SIZE)
                     .batch(BATCH_SIZE, drop_remainder=True))
    dataset_dev = (tf.data.Dataset.from_tensor_slices({'en': dev_token_en,
                                                       'jp': dev_token_jp}).batch(BATCH_SIZE, drop_remainder=True))

    dataset_test = (tf.data.Dataset.from_tensor_slices({'en': test_token_jp,
                                                        'jp': test_token_jp}).batch(BATCH_SIZE, drop_remainder=True))

    return dataset_train, dataset_dev, dataset_test, en_token, jp_token
