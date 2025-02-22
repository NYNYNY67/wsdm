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
)

from wsdm.preprocess import (
    render_user_prompt,
    get_chat_conversation,
    apply_chat_template,
)
from wsdm.causal_lm_infer import infer
from wsdm.postprocess import postprocess
from wsdm.evaluate import evaluate


@hydra.main(version_base=None, config_path="conf", config_name="zero_shot_causal_lm")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_parquet(data_dir / "original" / "train.parquet")

    if cfg.debug:
        df_train = df_train.sample(100)
        logger.warning("Debug mode is on. Only a subset of the data will be used.")

    logger.info(f"device: {cfg.device}")
    logger.info(f"model: {cfg.model}")

    if cfg.device == "cuda":
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    logger.info(f"torch_dtype: {torch_dtype}")

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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.device,
        use_cache=True,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
    )

    logger.info("Preprocessing the training data...")
    df_train = render_user_prompt(df_train)
    df_train = get_chat_conversation(df_train)
    df_train = apply_chat_template(df_train, tokenizer)

    logger.info("Inference on the training data...")
    df_train = infer(
        df=df_train,
        model=model,
        tokenizer=tokenizer,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )

    logger.info("Postprocessing the result...")
    df_train = postprocess(df_train)

    logger.info("Saving the result...")
    df_train.to_parquet(out_dir / "train.parquet")

    logger.info("Evaluating the model...")
    evaluate(df_train)


if __name__ == "__main__":
    main()
