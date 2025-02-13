import pathlib
from logzero import logger
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from wsdm.preprocess import preprocess
from wsdm.causal_lm_infer import infer


@hydra.main(version_base=None, config_path="conf", config_name="causal_lm_zero_shot")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_parquet(data_dir / "original" / "train.parquet")

    if cfg.debug:
        df_train = df_train.sample(100)
        logger.warning("Debug mode is on. Only a subset of the data will be used.")

    logger.info(f"device: {cfg.device}")
    logger.info(f"model: {cfg.model}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.device,
    )

    df_train = preprocess(df_train, tokenizer)

    df_train = infer(
        df=df_train,
        model=model,
        tokenizer=tokenizer,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )

    df_train.to_parquet(out_dir / "train.parquet")


if __name__ == "__main__":
    main()
