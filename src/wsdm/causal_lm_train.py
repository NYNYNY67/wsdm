from logzero import logger
import pandas as pd
import torch
from torch.optim import AdamW
# from bitsandbytes.optim import PagedAdam8bit as AdamW
from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
)
from tqdm import tqdm

from wsdm.causal_lm_dataloader import get_dataloader, get_collator


def train(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    model: AutoPeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    epochs: int,
    lr: float,
    eval_steps: int,
    saturation_rounds: int,
):
    collator = get_collator(tokenizer)
    train_loader = get_dataloader(df_train, tokenizer, collator)
    valid_loader = get_dataloader(df_valid, tokenizer, collator)

    optimizer = AdamW(model.parameters(), lr=lr)

    list_train_loss = []
    list_valid_loss = []
    n_train_samples = 0
    train_loss = 0
    steps = 0
    best_loss = float("inf")
    best_model = None
    best_step = 0
    rounds_since_last_best_model = 0
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"train epoch: {epoch}"):
            model.train()
            loss = train_step(device, batch, model, optimizer)

            n_train_samples += 1
            train_loss += loss

            steps += 1
            if steps % eval_steps == 0:
                list_train_loss.append(train_loss / n_train_samples)
                logger.info(f"epoch: {epoch}, steps: {steps}, train_loss: {list_train_loss[-1]}")
                n_train_samples = 0
                train_loss = 0

                model.eval()
                n_valid_samples = 0
                valid_loss = 0
                for batch in tqdm(valid_loader, desc=f"eval epoch: {epoch}"):
                    loss = valid_step(device, batch, model)

                    n_valid_samples += 1
                    valid_loss += loss

                list_valid_loss.append(valid_loss / n_valid_samples)
                logger.info(f"epoch: {epoch}, steps: {steps}, valid_loss: {list_valid_loss[-1]}")

                if list_valid_loss[-1] < best_loss:
                    best_loss = list_valid_loss[-1]
                    best_step = steps
                    best_model = model.state_dict()
                    logger.info(f"best model updated, valid_loss: {best_loss}")

                    rounds_since_last_best_model = 0
                else:
                    rounds_since_last_best_model += 1

                    if rounds_since_last_best_model >= saturation_rounds:
                        logger.info(f"early stopping at step: {steps}, best_step: {best_step}")
                        model.load_state_dict(best_model)
                        return {
                            "train_loss": list_train_loss,
                            "valid_loss": list_valid_loss,
                            "model": model,
                        }

    logger.info(f"training completed, best_step: {best_step}")
    model.load_state_dict(best_model)

    return {
        "train_loss": list_train_loss,
        "valid_loss": list_valid_loss,
        "model": model,
    }


def train_step(
    device: str,
    batch: dict,
    model: AutoPeftModelForCausalLM,
    optimizer: torch.optim.Optimizer,
):
    batch = {k: v.to(device) for k, v in batch.items()}
    model.train()
    output = model(**batch, logits_to_keep=10)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def valid_step(
    device: str,
    batch: dict,
    model: AutoPeftModelForCausalLM,
):
    batch = {k: v.to(device) for k, v in batch.items()}
    model.eval()
    output = model(**batch, logits_to_keep=10)
    loss = output.loss
    return loss.item()