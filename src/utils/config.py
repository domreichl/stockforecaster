import yaml
from pathlib import Path


class Config:
    def __init__(self, file_name="config.yaml"):
        cfg = load_yaml_file(file_name)
        self.ISIN = str(cfg["ISIN"])
        self.index = str(cfg["index"])
        self.train_days = int(cfg["train_days"])
        self.eval_days = int(cfg["eval_days"])
        self.holding_days = int(cfg["holding_days"])
        self.buy_threshold = float(cfg["buy_threshold"])
        self.features = list(cfg["features"])
        self.hparams = dict(cfg["hyperparameters"])
        self.look_back_window = self.hparams["look_back_window"]


class TuningParameters:
    def __init__(self, file_name="tuning.yaml"):
        params = load_yaml_file(file_name)
        self.replications = int(params["replications"])
        self.hparams = {k: list(v) for k, v in params.items() if k != "replications"}


def load_yaml_file(file_name: str) -> dict:
    path = Path(__file__).parent.parent.parent / file_name
    content = yaml.safe_load(open(path, "r"))
    return content
