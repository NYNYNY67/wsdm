import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)
from datasets import Dataset


def get_dataloader(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
):
    texts = df["text"].values.tolist()
    labels = df["winner"].map({"model_a": 0, "model_b": 1}).values.tolist()

    dataset = Dataset.from_dict({"text": texts, "labels": labels})
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
        columns=["input_ids", "attention_mask", "labels"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
    )
    return dataloader
