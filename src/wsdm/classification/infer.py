import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import Dataset
from tqdm import tqdm


def infer(
    df: pd.DataFrame,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
):
    texts = df["text"].values.tolist()

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            padding=False,
            truncation=False,
            padding_side="left",
        ),
        batched=False,
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    dataloader = DataLoader(dataset, batch_size=1)

    preds = []
    for batch in tqdm(dataloader, desc="classification inference"):
        batch = {k: v.squeeze(0).to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits
        preds.append(logits.argmax(1).item())

    df["pred"] = preds
    df["pred"] = df["pred"].map({0: "model_a", 1: "model_b"})
    return df
