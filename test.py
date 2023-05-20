from preprocessing import data_loader, data_builder, get_processors
import constants
import tensorflow as tf

if constants.IS_REBUILD == 1:
    train, dev, test = data_builder(constants.PICKLE + 'dataset.pkl')
else:
    train, dev, test = data_loader(constants.PICKLE + 'dataset.pkl')

en_processor, jp_processor, process_text = get_processors(train)

train.map(process_text, tf.data.AUTOTUNE)
for (ex_context_tok, ex_tar_in), ex_tar_out in train.take(1):
    print(ex_context_tok[0, :10].numpy())
    print()
    print(ex_tar_in[0, :10].numpy())
    print(ex_tar_out[0, :10].numpy())
