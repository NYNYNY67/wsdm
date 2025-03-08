import pathlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
)

from wsdm.trainer import Trainer


class CausalLmTrainer(Trainer):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        model: AutoPeftModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        epochs: int,
        lr: float,
        eval_steps: int,
        saturation_rounds: int,
        save_dir: pathlib.Path,
        early_stopping_criterion: str,
        larger_is_better: bool,
    ):
        super().__init__(
            df_train=df_train,
            df_valid=df_valid,
            model=model,
            tokenizer=tokenizer,
            device=device,
            epochs=epochs,
            lr=lr,
            eval_steps=eval_steps,
            saturation_rounds=saturation_rounds,
            save_dir=save_dir,
            early_stopping_criterion=early_stopping_criterion,
            larger_is_better=larger_is_better,
        )

    def get_dataloader(self, df):
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
            lambda x: self.tokenizer(
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

    def train_step(self, batch: dict):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        self.model.train()
        output = self.model(**batch, logits_to_keep=1)

        loss = output.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_iteration += 1
        self.n_iteration_in_round += 1

        labels = batch["labels"][:, -1]
        preds = output.logits.argmax(dim=-1)[:, -2]

        accuracy = (labels == preds).float().mean()

        return loss.item(), accuracy.item()

    def valid_step(self, batch: dict):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(**batch, logits_to_keep=1)
        loss = output.loss

        labels = batch["labels"][:, -1]
        preds = output.logits.argmax(dim=-1)[:, -2]

        accuracy = (labels == preds).float().mean()
        return loss.item(), accuracy.item()
