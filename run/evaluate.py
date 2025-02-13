from logzero import logger
from omegaconf import DictConfig
import hydra
import pandas as pd


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig):
    df_eval = pd.read_parquet(cfg.eval_file_path)

    logger.info(f"eval_file_path: {cfg.eval_file_path}")
    print(df_eval.head())

    cnt_model_a = df_eval["winner"].value_counts().get("model_a", 0)
    cnt_model_b = df_eval["winner"].value_counts().get("model_b", 0)

    logger.info(f"winner, model_a: {cnt_model_a}, model_b: {cnt_model_b}")

    cnt_model_a = df_eval["pred"].value_counts().get("model_a", 0)
    cnt_model_b = df_eval["pred"].value_counts().get("model_b", 0)

    logger.info(f"prediction, model_a: {cnt_model_a}, model_b: {cnt_model_b}")

    accuracy = (df_eval["winner"] == df_eval["pred"]).mean()
    logger.info(f"accuracy: {accuracy}")


if __name__ == "__main__":
    main()
