import pathlib
from logzero import logger
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from wsdm.preprocess import (
    render_templates,
    get_chat_conversation,
    apply_chat_template,
)
from wsdm.cross_validation import cross_validation
from wsdm.causal_lm_train import train


@hydra.main(version_base=None, config_path="conf", config_name="causal_lm_train")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_parquet(data_dir / "original" / "train.parquet")

    if cfg.debug:
        df_train = df_train.sample(100)
        cfg.epochs = 10
        logger.warning("Debug mode is on. Only a subset of the data will be used.")

    logger.info(f"device: {cfg.device}")
    logger.info(f"model: {cfg.model}")

    if cfg.quantization.enabled:
        if cfg.quantization.n_bit == 4:
            load_in_8bit = False
            load_in_4bit = True
        elif cfg.quantization.n_bit == 8:
            load_in_8bit = True
            load_in_4bit = False
        else:
            raise ValueError("n_bit must be 4 or 8.")
    else:
        load_in_8bit = False
        load_in_4bit = False

    if cfg.device == "cuda":
        if load_in_8bit or not torch.cuda.is_bf16_supported():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    logger.info(f"torch_dtype: {torch_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    logger.info("Preprocessing the training data...")
    df_train = render_templates(df_train, with_answer=True)
    df_train = get_chat_conversation(df_train)
    df_train = apply_chat_template(df_train, tokenizer)
    df_train = df_train[[
        "id",
        "text",
        "winner",
    ]].copy().reset_index(drop=True)
    df_train = cross_validation(df_train, cfg.cross_validation.n_folds, cfg.cross_validation.random_state)

    for fold in range(cfg.cross_validation.n_folds):
        logger.info(f"Training fold: {fold}")

        df_train_fold = df_train[df_train["fold"] != fold]
        df_valid_fold = df_train[df_train["fold"] == fold]

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=cfg.device,
            use_cache=True,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
        )

        result = train(
            df_train=df_train_fold,
            df_valid=df_valid_fold,
            model=model,
            tokenizer=tokenizer,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )

        result["model"].save_pretrained(out_dir / f"model_fold_{fold}")
        tokenizer.save_pretrained(out_dir / f"model_fold_{fold}")

        if cfg.debug:
            break


if __name__ == "__main__":
    main()
