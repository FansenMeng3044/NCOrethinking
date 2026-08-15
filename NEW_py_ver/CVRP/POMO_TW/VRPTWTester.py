from POMO.CVRPTester import CVRPTester

from POMO_TW.VRPTWEnv import VRPTWEnv
from POMO_TW.VRPTWModel import VRPTWModel


class VRPTWTester(CVRPTester):
    ENV_CLASS = VRPTWEnv
    MODEL_CLASS = VRPTWModel
