import constants
import numpy as np
import evaluate
import os
from preprocessing import get_dataset
from transformers.keras_callbacks import KerasMetricCallback
from transformers import AdamWeightDecay

metric = evaluate.load("sacrebleu")


class ENtoJPModel:
    def __init__(self, encoder, tokenizer):
        self.model = encoder
        self.tokenizer = tokenizer
        self.model_path = constants.TRAINED_MODEL
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _add_data(self, data_train, data_test):
        self.data_train = get_dataset(data_train, self.model, self.tokenizer)
        self.data_test = get_dataset(data_test, self.model, self.tokenizer)

    @staticmethod
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def _add_train_ops(self):
        self.optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        self.model.compile(optimizer=self.optimizer)
        self.callback = KerasMetricCallback(metric_fn=self.compute_metrics)
        print(self.model.summary())

    def _train(self):
        self.model.fit(self.data_train, validation_split=0.1, batch=constants.BATCH_SIZE, epochs=constants.EPOCHS,
                       callbacks=[self.callback])
        self.model.save_weights(self.model_path)

    def build(self, train_data, test_data, training=None):
        self._add_data(train_data, test_data)
        self._add_train_ops()
        if training:
            self._train()
