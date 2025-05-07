import pandas as pd                      
from transformers import (              
    AutoTokenizer,                       
    AutoModelForCausalLM,               
    Trainer,                             
    TrainingArguments,                    
    DataCollatorForLanguageModeling    
)
from datasets import Dataset, DatasetDict 

tok = AutoTokenizer.from_pretrained("facebook/opt-350m") 
tok.pad_token = tok.eos_token  
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m") 
model.config.pad_token_id = model.config.eos_token_id 

train_df = pd.read_excel("train.xlsx")[["text"]]  
valid_df = pd.read_excel("valid.xlsx")[["text"]]  

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False), 
    "validation": Dataset.from_pandas(valid_df, preserve_index=False), 
})

def preprocess(examples):
    """Convert text to tokens (numbers) that the model can understand"""
    return tok(
        examples["text"],           
        truncation=True,           
        padding="max_length",       
        max_length=128,          
    )

ds = ds.map(preprocess, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(
    tokenizer=tok,  
    mlm=False       
)

trainer = Trainer(
    model=model,                    
    args=TrainingArguments(
        output_dir="./ft",            
        num_train_epochs=5,         
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        save_steps=1000,           
        save_total_limit=1,           
        logging_steps=100,          
        fp16=True,                   
    ),
    train_dataset=ds["train"],       
    eval_dataset=ds["validation"],   
    data_collator=collator,         
)

trainer.train()

trainer.save_model("./ft")          
tok.save_pretrained("./ft")          

print("\n\n\n\n\nDONE")