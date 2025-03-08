import pathlib
from logzero import logger
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd


@hydra.main(version_base=None, config_path="conf", config_name="classification_infer")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
