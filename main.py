import constants
from preprocessing import data_builder, data_loader, get_processors
from model import Translator
import tensorflow as tf
from utils import masked_acc, masked_loss


def main():
    if constants.IS_REBUILD == 1:
        train, dev, test = data_builder(constants.PICKLE + 'dataset.pkl')
    else:
        train, dev, test = data_loader(constants.PICKLE + 'dataset.pkl')

    with tf.device("/GPU:0"):
        en_processor, jp_processor, train, dev, test = get_processors(train, dev, test)
        vocab_size = 1.0 * jp_processor.vocabulary_size()
        val_step_per_epoch = vocab_size // constants.BATCH_SIZE
        model = Translator(units=constants.rnn_units, context_text_processor=en_processor,
                           target_text_processor=jp_processor)
        model.compile(optimizer='adam',
                      loss=masked_loss,
                      metrics=[masked_acc, masked_loss])
        model.fit(
            train,
            epochs=constants.EPOCHS,
            steps_per_epoch=constants.steps_per_epoch,
            validation_data=dev,
            validation_steps=val_step_per_epoch,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=constants.PATIENCE)])
        print(model.summary())


if __name__ == '__main__':
    main()
