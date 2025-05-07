# 1. Imports - These are the necessary libraries
import pandas as pd                       # For handling Excel data
from transformers import (                # HuggingFace's library for working with language models
    AutoTokenizer,                        # Handles text conversion to tokens the model can process
    AutoModelForCausalLM,                 # Loads pre-trained language models
    Trainer,                              # Handles the training process
    TrainingArguments,                    # Contains settings for training
    DataCollatorForLanguageModeling       # Prepares batches of data for language modeling
)
from datasets import Dataset, DatasetDict # For organizing and processing the data

# 2. Load the pre-trained model and tokenizer
# Facebook's OPT-350M - excellent balance of prediction quality and efficiency
tok = AutoTokenizer.from_pretrained("facebook/opt-350m")  # Load the tokenizer
tok.pad_token = tok.eos_token  # Fix: Set padding token to be the end-of-sequence token
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")  # Load the model
model.config.pad_token_id = model.config.eos_token_id  # Tell the model to use EOS as padding

# 3. Load data from Excel files
train_df = pd.read_excel("train.xlsx")[["text"]]  # Load training data, keep only the 'text' column
valid_df = pd.read_excel("valid.xlsx")[["text"]]  # Load validation data, keep only the 'text' column

# Convert pandas DataFrames to HuggingFace Datasets format
ds = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),  # Training dataset
    "validation": Dataset.from_pandas(valid_df, preserve_index=False),  # Validation dataset
})

# 4. Tokenize the text data (convert text to numbers)
def preprocess(examples):
    """Convert text to tokens (numbers) that the model can understand"""
    return tok(
        examples["text"],           # The text to tokenize
        truncation=True,            # Cut off text that's too long
        padding="max_length",       # Add padding tokens to make all samples same length
        max_length=128,             # Maximum sequence length
    )

# Apply the tokenization to all examples in the dataset
ds = ds.map(preprocess, batched=True, remove_columns=["text"])

# 5. Prepare data for language modeling
collator = DataCollatorForLanguageModeling(
    tokenizer=tok,  # The tokenizer to use
    mlm=False       # Not using masked language modeling (we want full next-token prediction)
)

# 6. Set up the trainer with training settings
trainer = Trainer(
    model=model,                      # The model to train
    args=TrainingArguments(
        output_dir="./ft",            # Directory to save model checkpoints
        num_train_epochs=5,           # Number of times to go through the entire dataset
        per_device_train_batch_size=4, # Balanced batch size for medium model
        gradient_accumulation_steps=1, # No need for gradient accumulation with this model size
        save_steps=1000,              # Save a checkpoint every 1000 steps
        save_total_limit=1,           # Only keep the most recent checkpoint
        logging_steps=100,            # Log stats every 100 steps
        fp16=True,                    # Use mixed precision training for better performance
    ),
    train_dataset=ds["train"],        # The training dataset
    eval_dataset=ds["validation"],    # The validation dataset
    data_collator=collator,           # Prepares batches for training
)

# 7. Run the fine-tuning process
trainer.train()

# 8. Save the fine-tuned model and tokenizer
trainer.save_model("./ft")            # Save the model
tok.save_pretrained("./ft")           # Save the tokenizer

print("\n\n\n\n\nDONE")