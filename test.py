from preprocessing import data_loader, data_builder, get_processors, process_data, preprocess_jp
import constants
import tensorflow as tf
import re


if constants.IS_REBUILD == 1:
    train, dev, test = data_builder(constants.PICKLE + 'dataset.pkl')
else:
    train, dev, test = data_loader(constants.PICKLE + 'dataset.pkl')

en_processor, jp_processor, train, dev, test = get_processors(train, dev, test)

# print(jp_processor.get_vocabulary())
# for (ex_context_tok, ex_tar_in), ex_tar_out in train.take(1):
#     print(ex_context_tok[0, :10].numpy())
#     print()
#     print(ex_tar_in[0, :10].numpy())
#     print(ex_tar_out[0, :10].numpy())
# data_test = process_data(constants.DEV)

# test_sample = tf.constant(data_test[1][10])
# print(preprocess_jp(test_sample))

