import pathlib
from pprint import pprint
from logzero import logger
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd

from wsdm.preprocess import (
    render_user_prompt,
    get_chat_conversation,
)
from wsdm.request_openai import get_completion
from wsdm.postprocess import postprocess
from wsdm.evaluate import evaluate


@hydra.main(version_base=None, config_path="conf", config_name="causal_lm_infer_openai")
def main(cfg: DictConfig):
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    df_train = pd.read_parquet(data_dir / "original" / "train.parquet")

    if cfg.debug:
        df_train = df_train.sample(100)
        logger.warning("Debug mode is on. Only a subset of the data will be used.")

    logger.info("Preprocessing the training data...")
    df_train = render_user_prompt(df_train)
    df_train = get_chat_conversation(df_train)

    logger.info("Getting the completion from OpenAI...")
    df_train = get_completion(df_train, cfg.model)

    logger.info("Postprocessing the result...")
    df_train = postprocess(df_train)

    logger.info("Evaluating the result...")
    evaluate(df_train)


if __name__ == "__main__":
    main()
