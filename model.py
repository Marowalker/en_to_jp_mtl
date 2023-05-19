import tensorflow as tf
from keras.layers import Input, Dense, LSTM, LSTMCell
from keras.layers import Embedding, Bidirectional
import tensorflow_addons as tfa
import keras.backend as K
import constants
from keras.losses import SparseCategoricalCrossentropy
import sacrebleu
import os


def loss_function(y_true, y_pred):
    # shape of y [batch_size, ty]
    # shape of y_pred [batch_size, Ty, output_vocab_size]
    sparsecategoricalcrossentropy = SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y_true, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y_true, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


def initialize_initial_state():
    return [tf.zeros((constants.BATCH_SIZE, constants.rnn_units)),
            tf.zeros((constants.BATCH_SIZE, constants.rnn_units))]


def bleu_score(y_true, y_pred):
    score = sacrebleu.corpus_bleu(y_pred, [y_true]).score / 100
    return score


class EncoderNetwork(tf.keras.Model):
    def __init__(self, input_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.encoder_embedding = Embedding(input_dim=input_vocab_size,
                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = LSTM(rnn_units, return_sequences=True,
                                     return_state=True)
        self.initial_state = initialize_initial_state()

    def call(self, inputs):
        embeddings = self.encoder_embedding(inputs)
        a, a_tx, c_tx = self.encoder_rnnlayer(embeddings, initial_state=self.initial_state)
        return a, a_tx, c_tx


# DECODER
def build_attention_mechanism(units, memory, MSL):
    """
    MSL : Memory Sequence Length
    """
    # return tfa.seq2seq.LuongAttention(units, memory = memory,
    #                                  memory_sequence_length = MSL)
    return tfa.seq2seq.BahdanauAttention(units, memory=memory,
                                         memory_sequence_length=MSL)


class DecoderNetwork(tf.keras.Model):
    def __init__(self, output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = Embedding(input_dim=output_vocab_size,
                                           output_dim=embedding_dims)
        self.dense_layer = Dense(output_vocab_size)
        self.decoder_rnncell = LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = \
            build_attention_mechanism(constants.dense_units, None, constants.BATCH_SIZE * [constants.MAX_LENGTH])
        self.rnn_cell = self.build_rnn_cell(constants.BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell,
                                                sampler=self.sampler,
                                                output_layer=self.dense_layer
                                                )

    # wrap decodernn cell
    def build_rnn_cell(self, batch_size):
        return tfa.seq2seq.AttentionWrapper(self.decoder_rnncell,
                                            self.attention_mechanism,
                                            attention_layer_size=constants.dense_units)

    def build_decoder_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs):
        output_batch = inputs[0]
        a, a_tx, c_tx = inputs[1]

        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:, :-1]  # ignore eos
        # compare logits with timestepped +1 version of decoder_input

        # Decoder Embeddings
        decoder_emb_inp = self.decoder_embedding(decoder_input)

        # Setting up decoder memory from encoder output
        # and Zero State for AttentionWrapperState
        self.attention_mechanism.setup_memory(a)
        decoder_initial_state = self.build_decoder_initial_state(constants.BATCH_SIZE,
                                                                 encoder_state=[a_tx, c_tx],
                                                                 Dtype=tf.float32)

        # BasicDecoderOutput
        outputs, _, _ = self.decoder(decoder_emb_inp, initial_state=decoder_initial_state,
                                     sequence_length=constants.BATCH_SIZE * [constants.MAX_LENGTH - 1])

        return outputs.rnn_output


class ENtoJPModel:
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dims, rnn_units):
        self.encoder = EncoderNetwork(input_vocab_size, embedding_dims, rnn_units)
        self.decoder = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units)
        self.model_path = constants.TRAINED_MODEL

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _add_inputs(self):
        self.input_en = Input(shape=(constants.MAX_LENGTH,), dtype=tf.int32)
        self.input_jp = Input(shape=(constants.MAX_LENGTH,), dtype=tf.int32)

    def _add_data(self, train_data, val_data, test_data):
        self.dataset_train = train_data
        self.dataset_val = val_data
        self.dataset_test = test_data

    def _encoder_decoder_layer(self):
        encoder_outputs = self.encoder(self.input_en)
        logits, output = self.decoder([self.input_jp, encoder_outputs])
        return logits, output

    def _add_model(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = tf.keras.Model(
            inputs=(self.input_en, self.input_jp),
            outputs=(self._encoder_decoder_layer())
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=loss_function,
            metrics=[bleu_score]
        )
        print(self.model.summary())

    def _train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_bleu_score', mode='max',
                                                          patience=constants.PATIENCE)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=True,
            monitor='val_bleu_score',
            mode='max',
            save_best_only=True)

        self.model.fit(x=self.dataset_train, y=self.dataset_train['jp'][:, 1:],
                       validation_data=(self.dataset_val, self.dataset_val['jp'][:, 1:]),
                       epochs=constants.EPOCHS,
                       batch_size=constants.BATCH_SIZE,
                       callbacks=[early_stopping, model_checkpoint_callback])

    def build(self, train_data, val_data, test_data, is_training=True):
        self._add_inputs()
        self._add_data(train_data, val_data, test_data)
        self._add_model()
        if is_training:
            self._train()
