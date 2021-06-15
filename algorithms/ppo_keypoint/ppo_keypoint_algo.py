import torch
import torchvision

from ..common.ppo.ppo_base_algo import UnsupPpoAlgo
from ..common.utils import preprocess_obs


class KeypointPpoAlgo(UnsupPpoAlgo):
    def train_unsup(self):
        obs_data = self.obs_buffer.sample(self.batch_size, env=None)

        obs_data = self.policy.process_multiview_obs(obs_data)
        obs_data = preprocess_obs(obs_data, self.observation_space)
        images = obs_data['images']
        u_shift = obs_data['u_shift'] if self.policy.offset_crop else None
        v_shift = obs_data['v_shift'] if self.policy.offset_crop else None
        first_frame = obs_data['first_frame'] if self.policy.first_frame is not None else None
        keypoints, unsup_loss_dict, _ = self.policy.unsup_net.encode(images, rsample=True,
                                                                     u_shift=u_shift, v_shift=v_shift)
        images_hat = self.policy.unsup_net.decode(keypoints, u_shift=u_shift, v_shift=v_shift, first_frame=first_frame)
        unsup_loss_dict['ae_loss'] = self.policy.ae_criteria(images_hat, images)

        unsup_loss_dict = {k: v.mean() for k, v in unsup_loss_dict.items()}
        unsup_loss = sum([v * self.unsup_coef_dict[k] for k, v in unsup_loss_dict.items()])

        self.policy.unsup_optimizer.zero_grad()
        unsup_loss.backward()
        unsup_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.unsup_net.parameters(),
            1.5
        ).item()
        self.policy.unsup_optimizer.step()

        return unsup_loss_dict, unsup_grad_norm

    def save_visualization(self, path, batch_size=16):
        obs_data = self.obs_buffer.sample(batch_size, env=None)
        vis_tensor = self.policy.visualize(obs_data)
        torchvision.utils.save_image(vis_tensor, path, 1, padding=5)

