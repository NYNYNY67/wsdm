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
from peft import PeftModel

from wsdm.preprocess import (
    render_templates,
    get_chat_conversation,
    apply_chat_template,
)
from wsdm.cross_validation import cross_validation
from wsdm.causal_lm.infer import infer
from wsdm.postprocess import postprocess
from wsdm.evaluate import evaluate


@hydra.main(version_base=None, config_path="conf", config_name="causal_lm_infer")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_parquet(data_dir / "original" / "train.parquet")
    df_train = cross_validation(df_train, cfg.cross_validation.n_folds, cfg.cross_validation.random_state)
    df_train = df_train[df_train["fold"] == cfg.fold].copy()

    if cfg.debug:
        df_train = df_train.sample(100)
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.device,
        use_cache=True,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config if cfg.quantization.enabled else None,
    )
    model = PeftModel.from_pretrained(
        model,
        model_id=pathlib.Path(cfg.peft_dir),
        device=cfg.device,
    )

    logger.info("Preprocessing the training data...")
    df_train = render_templates(
        df_train,
        with_answer=True,
        response_max_length=cfg.preprocess.response_max_length,
        query_max_length=cfg.preprocess.query_max_length,
    )
    df_train = get_chat_conversation(df_train)
    df_train = apply_chat_template(df_train, tokenizer)
    df_train = df_train[[
        "id",
        "text",
        "winner",
    ]].copy()

    logger.info("Inference on the training data...")
    df_train = infer(
        df=df_train,
        model=model,
        tokenizer=tokenizer,
        device=cfg.device,
    )

    logger.info("Postprocessing the result...")
    df_train = postprocess(df_train)

    logger.info("Saving the result...")
    df_train.to_parquet(out_dir / "train.parquet")

    logger.info("Evaluating the model...")
    evaluate(df_train)


if __name__ == "__main__":
    main()
