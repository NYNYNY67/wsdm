import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from tqdm import tqdm


def infer(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
):
    texts = df["text"].values.tolist()

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ),
        batched=True,
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    responses = []
    for batch in tqdm(dataloader, desc="causal lm inference"):
        batch = {k: v.to(device) for k, v in batch.items()}
        generated_ids = model.generate(
            **batch,
            max_new_tokens=10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(batch["input_ids"], generated_ids)
        ]
        responses.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    df["response"] = responses

    return df
