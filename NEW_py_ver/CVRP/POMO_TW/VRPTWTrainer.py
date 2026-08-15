from POMO.CVRPTrainer import CVRPTrainer

from POMO_TW.VRPTWEnv import VRPTWEnv
from POMO_TW.VRPTWModel import VRPTWModel


class VRPTWTrainer(CVRPTrainer):
    ENV_CLASS = VRPTWEnv
    MODEL_CLASS = VRPTWModel
    METRICS_NAME = "pomo_cvrptw"
