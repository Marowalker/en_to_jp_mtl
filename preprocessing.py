import constants
from transformers import DataCollatorForSeq2Seq

source_lang = 'original_en'
target_lang = 'simplified_ja'
prefix = "translate English to Japanese: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples]
    targets = [example[target_lang] for example in examples]
    model_inputs = constants.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


def get_train_test(ratio=0.2):
    data = constants.data['train']
    n_sample = int(len(data['original_en']) * ratio)

    # all features in the dataset
    props = ['ID', 'original_ja', 'simplified_ja', 'original_en']

    # create train and test datasets
    train_data, test_data = {}, {}
    for prop in props:
        train_data[prop] = data[prop][:n_sample]
        test_data[prop] = data[prop][n_sample:]
    return train_data, test_data


tr, tt = get_train_test()
print(tr['simplified_ja'][:10])
