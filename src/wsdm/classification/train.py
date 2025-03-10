import pathlib
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
)

from wsdm.trainer import Trainer


class ClassificationTrainer(Trainer):
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
        is_accelerate_enabled: bool,
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
            is_accelerate_enabled=is_accelerate_enabled,
        )

    def get_dataloader(self, df):
        texts = df["text"].values.tolist()
        labels = df["winner"].map({"model_a": 0, "model_b": 1}).values.tolist()

        dataset = Dataset.from_dict({"text": texts, "labels": labels})
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
            columns=["input_ids", "attention_mask", "labels"],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
        )
        return dataloader

    def train_step(self, batch: dict):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        self.model.train()
        output = self.model(**batch)

        loss = output.loss

        if self.is_accelerate_enabled:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.n_iteration += 1
        self.n_iteration_in_round += 1

        labels = batch["labels"]
        preds = output.logits.argmax(dim=-1)

        accuracy = (labels == preds).float().mean()

        return loss.item(), accuracy.item()

    def valid_step(self, batch: dict):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(**batch)
        loss = output.loss

        labels = batch["labels"]
        preds = output.logits.argmax(dim=-1)

        accuracy = (labels == preds).float().mean()
        return loss.item(), accuracy.item()
