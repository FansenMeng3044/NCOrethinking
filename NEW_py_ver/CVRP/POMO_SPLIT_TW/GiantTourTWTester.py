from POMO.CVRPTester import CVRPTester
from POMO_SPLIT.GiantTourModel import GiantTourModel
from POMO_SPLIT_TW.GiantTourTWEnv import GiantTourTWEnv


class GiantTourTWTester(CVRPTester):
    ENV_CLASS = GiantTourTWEnv
    MODEL_CLASS = GiantTourModel
