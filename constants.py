from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from datasets import load_dataset


encoder = TFAutoModelForSeq2SeqLM.from_pretrained('t5-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('t5-base')

data = load_dataset('snow_simplified_japanese_corpus', 'snow_t15')

TRAINED_MODEL = 'trained_models/'

EPOCHS = 5

BATCH_SIZE = 16
