from stable_baselines3.common.callbacks import BaseCallback
import os


class SaveObsbufferCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name='obs_buffer', verbose=0):
        super(SaveObsbufferCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name = name

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, self.name)
            self.model.save_obs_buffer(path)
            if self.verbose > 1:
                print(f"Saving obs buffer to {path}")
        return True


class VisualizeCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='sample_images', verbose=0):
        super(VisualizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f'{self.name_prefix}_{self.num_timesteps}.jpg')
            self.model.save_visualization(path)
            if self.verbose > 1:
                print(f"Saving sample images to {path}")
        return True
