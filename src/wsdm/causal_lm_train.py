from logzero import logger
import pandas as pd
from torch.optim import AdamW
# from bitsandbytes.optim import PagedAdam8bit as AdamW
from transformers import (
    AutoTokenizer,
)
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
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
):
    collator = get_collator(tokenizer)
    train_loader = get_dataloader(df_train, tokenizer, collator)
    valid_loader = get_dataloader(df_valid, tokenizer, collator)

    optimizer = AdamW(model.parameters(), lr=lr)

    list_train_loss = []
    list_valid_loss = []
    for epoch in range(epochs):
        model.train()
        n_train_samples = 0
        epoch_train_loss = 0
        for batch in tqdm(train_loader, desc=f"train epoch: {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.train()
            output = model(**batch, logits_to_keep=10)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            n_train_samples += 1
            epoch_train_loss += loss.item()

        list_train_loss.append(epoch_train_loss / n_train_samples)
        logger.info(f"epoch: {epoch}, train_loss: {list_train_loss[-1]}")

        model.eval()
        n_valid_samples = 0
        epoch_valid_loss = 0
        for batch in tqdm(valid_loader, desc=f"valid epoch: {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.eval()
            output = model(**batch, logits_to_keep=10)
            loss = output.loss

            n_valid_samples += 1
            epoch_valid_loss += loss.item()

        list_valid_loss.append(epoch_valid_loss / n_valid_samples)
        logger.info(f"epoch: {epoch}, valid_loss: {list_valid_loss[-1]}")

    return {
        "train_loss": list_train_loss,
        "valid_loss": list_valid_loss,
        "model": model,
    }
