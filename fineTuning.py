# finetune_and_generate.py

# 1. Imports
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, DatasetDict

# 2. Load & fix tokenizer/model
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model.config.pad_token_id = model.config.eos_token_id

# 3. Load Excel into HF Datasets
train_df = pd.read_excel("train.xlsx")[["text"]]
valid_df = pd.read_excel("valid.xlsx")[["text"]]
ds = DatasetDict({
    "train":      Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(valid_df, preserve_index=False),
})

# 4. Tokenization
def preprocess(examples):
    return tok(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

ds = ds.map(preprocess, batched=True, remove_columns=["text"])

# 5. Data collator
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# 6. Trainer setup
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./ft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=1,
        logging_steps=100,
    ),
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
)

# 7. Fine-tune
trainer.train()

# 8. Save fine-tuned model & tokenizer
trainer.save_model("./ft")
tok.save_pretrained("./ft")

# 9. Load fine-tuned model for generation
tok2 = AutoTokenizer.from_pretrained("./ft")
tok2.pad_token = tok2.eos_token
m2 = AutoModelForCausalLM.from_pretrained("./ft")
m2.config.pad_token_id = m2.config.eos_token_id

# 10. Generate from prompt
inputs = tok2("In 2013, crime statistics showed", return_tensors="pt")
outputs = m2.generate(
    **inputs,
    max_length=128 + 60,
    do_sample=True,
    top_p=0.9,
    pad_token_id=m2.config.pad_token_id
)
print(tok2.decode(outputs[0], skip_special_tokens=True))
