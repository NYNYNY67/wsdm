import pathlib
from logzero import logger
from omegaconf import OmegaConf
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
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
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
        cfg.epochs = 1
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

    logger.info(f"quantization: {cfg.quantization.enabled}")
    logger.info(f"load_in_8bit: {load_in_8bit}")
    logger.info(f"load_in_4bit: {load_in_4bit}")

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
    peft_config = LoraConfig(
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        r=cfg.lora.r,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=OmegaConf.to_container(cfg.lora.target_modules),
        inference_mode=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    logger.info("Preprocessing the training data...")
    df_train = render_templates(df_train, with_answer=True, response_max_length=cfg.preprocess.response_max_length)
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

        if cfg.validation_data_size:
            df_valid_fold = df_valid_fold.sample(cfg.validation_data_size, random_state=cfg.cross_validation.random_state).reset_index(drop=True)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map="auto",
            attn_implementation=cfg.attn_implementation,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            use_cache=False,
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": True})
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        result = train(
            df_train=df_train_fold,
            df_valid=df_valid_fold,
            model=model,
            tokenizer=tokenizer,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            eval_steps=cfg.eval_steps,
            saturation_rounds=cfg.saturation_rounds,
        )

        result["model"].save_pretrained(out_dir / f"model_fold_{fold}")
        tokenizer.save_pretrained(out_dir / f"model_fold_{fold}")

        if cfg.debug:
            break


if __name__ == "__main__":
    main()
