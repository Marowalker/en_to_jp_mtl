import constants
from transformers import DataCollatorForSeq2Seq
import tensorflow as tf

source_lang = 'original_en'
target_lang = 'simplified_ja'
prefix = "translate English to Japanese: "


def preprocess_function(examples):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    model_inputs = constants.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


def get_train_test(ratio=0.1):
    data = constants.data['train'].train_test_split(test_size=ratio)
    train_data, test_data = data['train'], data['test']
    return train_data, test_data


def get_dataset(data, model, tokenizer):
    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
    return model.prepare_tf_dataset(
        tokenized_data,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

# data = get_train_test()
# tokenized_data = data['test'].map(preprocess_function, batched=True)
# data_collator = DataCollatorForSeq2Seq(tokenizer=constants.tokenizer, model=constants.encoder, return_tensors="tf")
# print(tokenized_data)
