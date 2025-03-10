import pathlib
from logzero import logger
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModel,
)
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        model: AutoPeftModel,
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
        self.df_train = df_train
        self.df_valid = df_valid
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.eval_steps = eval_steps
        self.saturation_rounds = saturation_rounds
        self.save_dir = save_dir
        self.early_stopping_criterion = early_stopping_criterion
        self.larger_is_better = larger_is_better
        self.is_accelerate_enabled = is_accelerate_enabled

        self.train_loader = self.get_dataloader(df_train)
        self.valid_loader = self.get_dataloader(df_valid)
        self.optimizer = AdamW(model.parameters(), lr=lr)

        self.best_loss = float("inf")
        self.best_metric = float("-inf") if larger_is_better else float("inf")
        self.log_steps = []
        self.train_losses = []
        self.valid_losses = []
        self.train_metrics = []
        self.valid_metrics = []

        self.n_iteration = 0
        self.rounds_since_last_best_model = 0
        self.n_iteration_in_round = 0
        self.train_loss_round = 0
        self.train_metric_round = 0

        self.is_early_stopped = False

        if is_accelerate_enabled:
            from accelerate import Accelerator

            self.accelerator = Accelerator()
            self.model, self.optimizer, self.train_loader, self.valid_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.valid_loader
            )
            self.device = self.accelerator.device

    def get_dataloader(self, df: pd.DataFrame):
        raise NotImplementedError()

    def train(self):
        for epoch in range(self.epochs):
            if self.is_early_stopped:
                break

            logger.info(f"epoch: {epoch}")
            self.train_epoch()

        self.plot_learning_log()

    def train_epoch(self):
        for batch in tqdm(self.train_loader, desc="train"):

            if self.is_early_stopped:
                break

            self.model.train()

            loss, metric = self.train_step(batch)

            self.train_loss_round += loss
            self.train_metric_round += metric

            del loss, metric

            if self.n_iteration % self.eval_steps == 0:
                self.if_eval_step()

    def if_eval_step(self):
        train_loss = self.train_loss_round / self.n_iteration_in_round
        train_metric = self.train_metric_round / self.n_iteration_in_round

        valid_loss, valid_metric = self.valid()

        logger.info(f"n_iteration: {self.n_iteration}")
        logger.info(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
        logger.info(f"train_metric: {train_metric}, valid_metric: {valid_metric}")

        self.log_steps.append(self.n_iteration)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.train_metrics.append(train_metric)
        self.valid_metrics.append(valid_metric)

        self.train_loss_round = 0
        self.n_iteration_in_round = 0
        self.train_metric_round = 0

        if self.early_stopping_criterion == "loss":
            if valid_loss < self.best_loss:
                self.if_best(valid_loss, valid_metric)
            else:
                self.if_not_best(valid_loss, valid_metric)
        elif self.early_stopping_criterion == "metric":
            if self.larger_is_better:
                if valid_metric > self.best_metric:
                    self.if_best(valid_loss, valid_metric)
                else:
                    self.if_not_best(valid_loss, valid_metric)
            else:
                if valid_metric < self.best_metric:
                    self.if_best(valid_loss, valid_metric)
                else:
                    self.if_not_best(valid_loss, valid_metric)
        else:
            raise ValueError(f"early_stopping_criterion: {self.early_stopping_criterion} is not supported")

    def if_best(self, valid_loss, valid_metric):
        self.best_loss = valid_loss
        self.best_metric = valid_metric
        self.model.save_pretrained(self.save_dir)
        logger.info(f"best model updated, valid_loss: {self.best_loss}, valid_metric: {self.best_metric}")

        self.rounds_since_last_best_model = 0

    def if_not_best(self, valid_loss, valid_metric):
        self.rounds_since_last_best_model += 1
        logger.info(f"model not updated, valid_loss: {valid_loss}, valid_metric: {valid_metric}")
        logger.info(f"best_loss: {self.best_loss}, best_metric: {self.best_metric}")

        if self.rounds_since_last_best_model >= self.saturation_rounds:
            logger.info(f"early stopping at step: {self.n_iteration}")
            self.is_early_stopped = True

    def train_step(self, batch: dict):
        raise NotImplementedError()

    def valid(self):
        self.model.eval()

        with torch.no_grad():
            n_valid_samples = 0
            valid_loss = 0
            valid_metric = 0
            for batch in tqdm(self.valid_loader, desc="valid"):
                loss, metric = self.valid_step(batch)

                n_valid_samples += 1
                valid_loss += loss
                valid_metric += metric

            valid_loss = valid_loss / n_valid_samples
            valid_metric = valid_metric / n_valid_samples
            return valid_loss, valid_metric

    def valid_step(self, batch: dict):
        raise NotImplementedError()

    def plot_learning_log(self):
        fig, ax = plt.subplots()
        ax.set_title("learning log (loss)")
        ax.plot(self.log_steps, self.train_losses, label="train")
        ax.plot(self.log_steps, self.valid_losses, label="valid")
        ax.legend()
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        fig.savefig(self.save_dir / "learning_log_loss.png")

        fig, ax = plt.subplots()
        ax.set_title("learning log (metric)")
        ax.plot(self.log_steps, self.train_metrics, label="train")
        ax.plot(self.log_steps, self.valid_metrics, label="valid")
        ax.legend()
        ax.set_xlabel("iteration")
        ax.set_ylabel("metric")
        fig.savefig(self.save_dir / "learning_log_metric.png")
