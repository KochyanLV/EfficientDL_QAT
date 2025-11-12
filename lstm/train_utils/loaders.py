from lstm.lstm_utils.load_dataset import load_data
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


def load_tokenizer(tokenizer_path: str = "../own_tokenizer"):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer
    
    
def tokenize_batch_no_pad(batch, tokenizer, max_len: int = 512):
    enc = tokenizer(
        batch["text"],
        padding=False,
        truncation=True,
        max_length=max_len,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": batch["label"],
    }
    
    
def make_loaders(tokenizer_path: str = "../own_tokenizer", batch_size: int = 8):
    
    dataset = load_data()
    tokenizer = load_tokenizer(tokenizer_path=tokenizer_path)
    
    proc = dataset.map(tokenize_batch_no_pad, batched=True, remove_columns=["text"])
    proc = proc.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    train_loader = DataLoader(proc["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_loader  = DataLoader(proc["test"],  batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    return train_loader, test_loader, tokenizer