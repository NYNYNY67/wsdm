from omegaconf import DictConfig
import hydra
import pandas as pd

from wsdm.evaluate import evaluate


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig):
    df_eval = pd.read_parquet(cfg.eval_file_path)
    evaluate(df_eval)


if __name__ == "__main__":
    main()
