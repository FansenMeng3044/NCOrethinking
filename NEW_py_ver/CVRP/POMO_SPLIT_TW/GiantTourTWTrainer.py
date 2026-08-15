from POMO_SPLIT.GiantTourModel import GiantTourModel
from POMO_SPLIT.GiantTourTrainer import GiantTourTrainer
from POMO_SPLIT_TW.GiantTourTWEnv import GiantTourTWEnv


class GiantTourTWTrainer(GiantTourTrainer):
    ENV_CLASS = GiantTourTWEnv
    MODEL_CLASS = GiantTourModel
    METRICS_NAME = "pomo_split_cvrptw"
