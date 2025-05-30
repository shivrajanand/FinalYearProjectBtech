from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datasets import Dataset
from transformers.integrations import TensorBoardCallback
import torch



# Loading the mode
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="san_Deva" , tgt_lang="hin_Deva")
print("TOKENIZER LOADED")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
print("MODEL LOADED")

# Freezing encoder to reduce model size
for param in model.model.encoder.parameters():
    param.requires_grad = False
    
print("ENCODER PARAMETERS FREEZED")

# for name, param in model.model.encoder.named_parameters():
#     if param.requires_grad:
#         print(f"{name} is trainable")
#     else:
#         print(f"{name} is frozen")

# Preparing the dataset
train_json = pd.read_json("train_file.json", lines=True)
train_df = pd.json_normalize(train_json["translation"])

# Dataset Creator
def tokenize_and_create_dataset(tokenizer, data_df, max_length=128):
    # Tokenize Sanskrit (source)
    encodings = tokenizer(
        list(data_df["sa"]),
        truncation=True,
        padding=True,
        max_length=max_length
    )

    # Tokenize Hindi (target)
    with tokenizer.as_target_tokenizer():
        decodings = tokenizer(
            list(data_df["hi"]),
            truncation=True,
            padding=True,
            max_length=max_length
        )

    # Create Dataset object
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": decodings["input_ids"],
    })

    return dataset

train_dataset = tokenize_and_create_dataset(tokenizer=tokenizer, data_df = train_df)
print("TRAINING DATASET CREATED")

test_json = pd.read_json("test_file.json", lines=True)
test_df = pd.json_normalize(train_json["translation"])
test_dataset = tokenize_and_create_dataset(tokenizer=tokenizer, data_df = test_df)
print("TESTED DATASET CREATED")

#Training

tensorboard_callback = TensorBoardCallback()
print("TENSORBOARD CALLBACK INITIATED")


try:
    torch.cuda.set_device(1)
except Exception as e:
    print(e)

if torch.cuda.current_device()!=1:
    print("CURRENT GPU: ", torch.cuda.current_device())
    exit(1)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding

print("TRAINING STARTS")

model_args = Seq2SeqTrainingArguments(
    output_dir=f"./output_dir",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=1000,  
    save_steps=5000,     
    save_total_limit=2,  
    evaluation_strategy="steps",
    eval_steps=5000,  
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.02,
    predict_with_generate=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=model_args,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[tensorboard_callback],

)

trainer.train()

torch.save(model.state_dict(), 'nllb_saTOhi_finetuned.pth')