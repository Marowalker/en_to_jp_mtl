import constants
from preprocessing import data_builder, data_loader, get_processors
from model import Translator
import tensorflow as tf
from utils import masked_acc, masked_loss
import os


def main():
    if constants.IS_REBUILD == 1:
        train, dev, test = data_builder(constants.PICKLE + 'dataset.pkl')
    else:
        train, dev, test = data_loader(constants.PICKLE + 'dataset.pkl')

    if not os.path.exists(constants.TRAINED_MODEL):
        os.makedirs(constants.TRAINED_MODEL)

    with tf.device("/GPU:0"):
        # comment this part (until the next comment) after training
        en_processor, jp_processor, train, dev, test = get_processors(train, dev, test)

        model = Translator(units=constants.rnn_units, context_text_processor=en_processor,
                           target_text_processor=jp_processor)
        model.compile(optimizer='adam',
                      loss=masked_loss,
                      metrics=[masked_acc, masked_loss])
        model.fit(
            train,
            epochs=constants.EPOCHS,
            validation_data=dev,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=constants.PATIENCE)])

        print(model.summary())
        model.save_weights(constants.TRAINED_MODEL)

        # After training, uncomment this and run again
        model.load_weights(constants.TRAINED_MODEL)
        example_sentence = input("Enter sample sentence: ")
        result = model.translate([example_sentence])
        print(result[0].numpy())


if __name__ == '__main__':
    main()
