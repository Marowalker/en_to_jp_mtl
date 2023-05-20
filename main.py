import constants
from preprocessing import data_builder, data_loader
# from model import ENtoJPModel
import tensorflow as tf


def main():
    if constants.IS_REBUILD == 1:
        train, dev, test = data_builder(constants.PICKLE + 'dataset.pkl')
    else:
        train, dev, test = data_loader(constants.PICKLE + 'dataset.pkl')

    # Vocab
    input_vocab_size = len(constants.en_tokenizer.word_index) + 1  # English
    output_vocab_size = len(constants.jp_tokenizer.word_index) + 1  # Japanese

    with tf.device("/GPU:0"):
        model = ENtoJPModel(input_vocab_size, output_vocab_size, constants.embedding_dims, constants.rnn_units)
        model.build(train, dev, test)


if __name__ == '__main__':
    main()
