from typing import Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM


def get_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
):
    def custom_collator(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_masks = torch.stack([item["attention_mask"] for item in batch])
        labels = input_ids.clone()
        labels[:, :-1] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

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
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collator,
    )
    return dataloader
