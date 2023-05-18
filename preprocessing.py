from constants import *


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

    for line in lines:
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


train_en, train_jp = process_data(TRAIN, is_train=True)
dev_en, dev_jp = process_data(DEV)
print(train_en)

# en_tokenizer.fit_on_texts(train_en)
# jp_tokenizer.fit_on_texts(train_jp)
#
# train_token_en = get_tokenized_data(train_en, en_tokenizer, is_train=True)
# train_token_jp = get_tokenized_data(train_jp, jp_tokenizer, is_train=True)
#
# dev_token_en = get_tokenized_data(dev_en, en_tokenizer)
# dev_token_jp = get_tokenized_data(dev_jp, jp_tokenizer)
#
# print(train_token_en)
# print(dev_token_jp)
