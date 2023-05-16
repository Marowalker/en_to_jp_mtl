import constants
from transformers import DataCollatorForSeq2Seq


tokenized_data = constants.data['train'].map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=constants.tokenizer, model=constants.encoder, return_tensors="tf")
print(data_collator)