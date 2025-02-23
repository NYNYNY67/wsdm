from typing import Optional
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM


def get_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    collate_fn: callable,
):
    texts = df["text"].values.tolist()

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            padding=False,
            truncation=False,
        ),
        batched=False,
    ).map(
        lambda x: {k: v[0] if isinstance(v, list) else v for k, v in x.items()},
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    return dataloader


def get_collator(
    tokenizer: AutoTokenizer,
):
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
    )
    return collator
