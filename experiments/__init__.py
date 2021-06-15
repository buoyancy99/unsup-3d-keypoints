from .common.parse import get_args
from .ppo_pixel.experiment import PpoPixelExperiment
from .ppo_rad.experiment import PpoRadExperiment
from .ppo_keypoint.experiment import PpoKeypointExperiment

algo_registry = dict(
    ppopixel=PpoPixelExperiment,
    pporad=PpoRadExperiment,
    ppokeypoint=PpoKeypointExperiment,
)

