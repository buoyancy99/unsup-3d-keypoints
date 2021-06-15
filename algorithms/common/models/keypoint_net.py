import torch
import torch.nn as nn
import numpy as np
import cv2

from .cnn import CustomeEncoder, CustomeDecoder
from utils.vis_utils import get_cmap

epsilon = 1e-8
DEBUG = 0


class KeypointNetBase(nn.Module):
    def __init__(
        self,
        observation_space,
        crop_size,
        projection_matrix,
        view_matrices,
        num_keypoints=12,
        encoder_cls=CustomeEncoder,
        decoder_cls=CustomeDecoder,
        latent_stack=False,
        n_filters=32,
        separation_margin=0.05,
        mean_depth=0.5,
        noise=0,
        independent_pred=True,
        decode_first_frame=False,
        decode_attention=False,
    ):
        super().__init__()
        frame_stack, self.num_cameras, self.original_size, _, self.image_channels = observation_space['images'].shape
        self.frame_stack = 1 if latent_stack else frame_stack
        self.image_size = self.original_size if crop_size is None else crop_size

        self.register_buffer("P", torch.from_numpy(projection_matrix).float())
        """
        Notice normalization matrix is multiplied into the view matrix for environments that enables renormalization. 
        However, when decoding, normalization and unnormalization cancels out, so no need to worry
        about decoding. Only the latent is affected (normalized).
        """
        self.register_buffer("Vs", torch.from_numpy(view_matrices).float())

        self.num_keypoints = num_keypoints * self.frame_stack
        self.encoder_cls = encoder_cls
        self.decoder_cls = decoder_cls
        self.latent_stack = latent_stack
        self.n_filters = n_filters
        self.separation_margin = separation_margin
        self.mean_depth = mean_depth
        self.noise = noise
        self.decode_first_frame = decode_first_frame
        self.decode_attention = decode_attention

        if independent_pred:
            self.groups = self.num_cameras
            self.n_filters = self.num_cameras * self.n_filters
        else:
            self.groups = 1

        self.first_frame_filters = self.n_filters // 2
        self.encoder = None
        self.first_frame_encoder = None
        self.decoder = None
        self.heatmap_size = None
        self.heatmap_head = None
        self.output_image_size = None
        self.upsampler = None

        self.build_model()

    def build_model(self):
        self.encoder = self.encoder_cls(self.num_cameras * self.frame_stack * self.image_channels,
                                        self.n_filters * 4,
                                        self.n_filters, groups=self.groups)
        if self.decode_first_frame:
            self.first_frame_encoder = self.encoder_cls(self.num_cameras * self.frame_stack * self.image_channels,
                                                        self.first_frame_filters * 4,
                                                        self.first_frame_filters, groups=self.groups)
        self.decoder = self.decoder_cls(self.num_cameras * self.num_keypoints +
                                        self.first_frame_filters * 4 * int(self.decode_first_frame),
                                        self.num_cameras * self.frame_stack * self.image_channels,
                                        self.n_filters, groups=self.groups)
        self.heatmap_head = nn.Conv2d(self.n_filters * 4, self.num_cameras * self.num_keypoints,
                                      1, 1, 0, groups=self.groups)

        self.heatmap_size = self.encoder.infer_output_size(self.image_size)
        self.output_image_size = self.decoder.infer_output_size(self.heatmap_size)
        self.encoder = nn.Sequential(self.encoder, nn.ReLU())

        r_y = torch.arange(0, self.heatmap_size, 1.0) / (self.heatmap_size - 1) * 2 - 1
        r_x = torch.arange(0, self.heatmap_size, 1.0) / (self.heatmap_size - 1) * 2 - 1
        rany, ranx = torch.meshgrid(-r_y, r_x)   # ranx left -1, right 1, rany top 1, bottom -1
        self.register_buffer("ranx", torch.FloatTensor(ranx).clone())
        self.register_buffer("rany", torch.FloatTensor(rany).clone())
        P_inv = torch.FloatTensor([[1.0 / self.P[0, 0], 0, 0, 0],
                                   [0, 1.0 / self.P[1, 1], 0, 0],
                                   [0, 0, -1.0 / self.P[2, 2], 0],
                                   [0, 0, 0, 1.0]])
        self.register_buffer("P_inv", P_inv)

        """
        Notice normalization matrix is multiplied into the view matrix, so inverse of V also contains inverse
        normalization. However, when decoding, normalization and unnormalization cancels out, so no need to worry
        about decoding. Only the latent is affected (normalized).
        """
        self.register_buffer("V_invs", torch.inverse(self.Vs))

        self.upsampler = nn.Upsample((self.output_image_size, self.output_image_size))

    def add_offset_xy(self, xy_normalized, u_shift, v_shift):
        """
        :param xy_normalized: (batch, num_cameras, num_keypoints, 2)
        :param u_shift: (batch, num_cameras)
        :param v_shift: (batch, num_cameras)
        :return: (batch, num_cameras, num_keypoints, 2)
        """
        if u_shift is None or v_shift is None:
            return xy_normalized
        else:
            x_shift = u_shift
            y_shift = 1 - v_shift - self.image_size / self.original_size
            shift = torch.stack([x_shift, y_shift], -1)[:, :, None]
            xy_normalized = (xy_normalized + 1) * self.image_size / self.original_size + 2 * shift - 1
            return xy_normalized

    def remove_offset_xy(self, xy_normalized, u_shift, v_shift):
        """
        :param xy_normalized: (batch, num_cameras, num_keypoints, 2)
        :param u_shift: (batch, num_cameras)
        :param v_shift: (batch, num_cameras)
        :return: (batch, num_cameras, num_keypoints, 2)
        """
        if u_shift is None or v_shift is None:
            return xy_normalized
        else:
            x_shift = u_shift
            y_shift = 1 - v_shift - self.image_size / self.original_size
            shift = torch.stack([x_shift, y_shift], -1)[:, :, None]
            xy_normalized = (xy_normalized + 1 - 2 * shift) * self.original_size / self.image_size - 1
            return xy_normalized

    def unproject(self, xy_normalized, z):
        xy = torch.cat([-xy_normalized * z, z, torch.ones_like(z)], 1)
        xyzw = torch.matmul(xy, self.P_inv.T)
        xyzw = xyzw / (xyzw[:, 3:4] + epsilon)
        return xyzw

    def project(self, xyzw):
        xyzw = torch.matmul(xyzw, self.P.T)
        xyzw = xyzw / (xyzw[:, 2:3] + epsilon)
        xy_normalized = xyzw[:, :2]
        return xy_normalized

    def heatmap_to_xy(self, heatmap):
        # notice uv and xy are different. xy are in cartesian coordinate
        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2), keepdim=True) + epsilon)
        sx = torch.sum(heatmap * self.ranx[None], dim=(1, 2))
        sy = torch.sum(heatmap * self.rany[None], dim=(1, 2))
        xy_normalized = torch.stack([sx, sy], 1)
        return xy_normalized

    def xy_to_heatmap(self, xy, scale, sigma=1.0):
        grid = torch.stack([self.ranx, self.rany], 2)[None]  # (b, hs, hs, 2)
        # sigma = torch.ones_like(sigma)
        if isinstance(sigma, float):
            var = (sigma / self.heatmap_size) ** 2.0
        else:
            # sigma (batch_size, 1)
            sigma = sigma[:, None]
            var = (sigma / self.heatmap_size) ** 2.0
        # heatmap = torch.exp(- torch.sum((grid - xy[:, None, None, :]) ** 2, 3) / (2 * var)) / (2 * np.pi * var)
        heatmap = torch.exp(- torch.sum((grid - xy[:, None, None, :]) ** 2, 3) / (2 * var))
        if self.decode_attention:
            heatmap = heatmap * scale[:, None] * self.num_keypoints / 2.0

        return heatmap

    def heatmap_to_scale(self, heatmap):
        """
        :param heatmap: (batch_size, heatmap_size, heatmap_size)
        :return:
        """
        scales = torch.mean(heatmap, dim=(1, 2)).unsqueeze(1)

        return scales

    def compute_z(self, heatmap, z_map):
        heatmap = heatmap + epsilon
        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2), keepdim=True) + epsilon)
        z = torch.sum(z_map * heatmap, dim=(1, 2)).unsqueeze(1)
        return z

    def xyz_world_to_cam(self, xyz_world):
        batch_size, num_points, _ = xyz_world.shape
        xyzw_world = torch.ones((batch_size, num_points, 4), dtype=xyz_world.dtype, device=xyz_world.device)
        xyzw_world[:, :, :3] = xyz_world
        xyzw_cam = torch.einsum('bjn,cmn->bcjm', xyzw_world, self.Vs)
        xyzw_cam = xyzw_cam / (xyzw_cam[:, :, :, 3:4] + epsilon)  # (batch, num_cam, num_points, 4)
        return xyzw_cam

    def xyz_world_to_xy_normalized(self, xyz_world):
        batch_size, num_points, _ = xyz_world.shape
        xyzw_cam = self.xyz_world_to_cam(xyz_world)
        xyzw_cam = xyzw_cam.reshape(batch_size * self.num_cameras * num_points, 4)
        xy_normalized = self.project(xyzw_cam).reshape(batch_size, self.num_cameras, num_points, 2)
        return xy_normalized

    def xyz_world_to_depth(self, xyz_world):
        batch_size, num_points, _ = xyz_world.shape
        xyzw_cam = self.xyz_world_to_cam(xyz_world)
        xyzw_cam = xyzw_cam.reshape(batch_size, self.num_cameras, num_points, 4)
        depth = xyzw_cam[..., 2:3]

        return depth

    def xyz_world_to_heatmap(self, xyz_world, u_shift, v_shift, scale=1.0):
        batch_size, num_points, _ = xyz_world.shape
        xy_normalized = self.xyz_world_to_xy_normalized(xyz_world)  # (batch_size, self.num_cameras, num_points, 2)
        xy_normalized = self.remove_offset_xy(xy_normalized, u_shift, v_shift)
        depth = self.xyz_world_to_depth(xyz_world)
        xy_normalized = xy_normalized.reshape(batch_size * self.num_cameras * num_points, 2)
        depth = depth.reshape(batch_size * self.num_cameras * num_points, 1)
        if not isinstance(scale, float):
            scale = scale[:, None].repeat(1, self.num_cameras, 1, 1)
            scale = scale.reshape(batch_size * self.num_cameras * num_points, 1)

        sigma = - 1.5 * self.mean_depth / depth
        if DEBUG:
            print('inv_depth mean {}, max {}, min {}'.format(sigma.mean().item(), sigma.max().item(), sigma.min().item()))
        heatmap = self.xy_to_heatmap(xy_normalized, scale, sigma)
        heatmap = heatmap.reshape(batch_size, self.num_cameras, num_points, self.heatmap_size, self.heatmap_size)
        return heatmap

    def forward(self, images, decode=False):
        keypoints, heatmap, unsup_loss_dict = self.encode(images)
        images_hat = None
        if decode:
            images_hat = self.decode(keypoints)
        return keypoints, heatmap, images_hat, unsup_loss_dict

    def encode(self, images, rsample=True, u_shift=None, v_shift=None):
        raise NotImplementedError

    def visualize(self, images, robot_joint_positions, robot_joints, keypoints=None):
        raise NotImplementedError

    def keypoints_to_heatmap(self, keypoints, u_shift, v_shift):
        raise NotImplementedError

    def decode(self, keypoints, u_shift=None, v_shift=None, first_frame=None):
        batch_size = keypoints.shape[0]
        keypoints_heatmap = self.keypoints_to_heatmap(keypoints, u_shift, v_shift)
        keypoints_heatmap = keypoints_heatmap.view(batch_size, -1, self.heatmap_size, self.heatmap_size)
        if self.decode_first_frame:
            first_frame_features = self.first_frame_encoder(first_frame * 2 - 1)
            latent = torch.cat([keypoints_heatmap, first_frame_features], 1)
        else:
            latent = keypoints_heatmap
        imgaes_hat = self.decoder(latent)
        imgaes_hat = imgaes_hat.reshape(batch_size, self.num_cameras * self.frame_stack * self.image_channels,
                                        self.output_image_size, self.output_image_size)

        imgaes_hat = imgaes_hat * 0.5 + 0.5

        return imgaes_hat

    def sparsity_loss(self, scales):
        loss = torch.sum(torch.abs(scales), (1, 2), keepdim=True)
        return loss


class KeypointNet3d(KeypointNetBase):
    def __init__(
            self,
            observation_space,
            crop_size,
            projection_matrix,
            view_matrices,
            num_keypoints,
            encoder_cls=CustomeEncoder,
            decoder_cls=CustomeDecoder,
            latent_stack=False,
            n_filters=32,
            separation_margin=0.05,
            mean_depth=0.5,
            noise=0,
            independent_pred=True,
            decode_first_frame=False,
            decode_attention=False
    ):

        self.z_head = None
        super(KeypointNet3d, self).__init__(
            observation_space,
            crop_size,
            projection_matrix,
            view_matrices,
            num_keypoints,
            encoder_cls=encoder_cls,
            decoder_cls=decoder_cls,
            latent_stack=latent_stack,
            n_filters=n_filters,
            separation_margin=separation_margin,
            mean_depth=mean_depth,
            noise=noise,
            independent_pred=independent_pred,
            decode_first_frame=decode_first_frame,
            decode_attention=decode_attention
        )

        self.cmap = get_cmap(self.num_keypoints)

    def build_model(self):
        super().build_model()
        self.z_head = nn.Conv2d(self.n_filters * 4, self.num_cameras * self.num_keypoints,
                                1, 1, 0, groups=self.groups)

    def weigh_with_confidence(self, xyz_world, scale):
        """
        :param xyz_world: (batch, num_cam, num_keypoints, 3)
        :param scale: (batch, num_cam, num_keypoints, 1)
        :return: (batch, num_keypoints, 3)
        """
        return torch.sum(xyz_world * (scale / torch.sum(scale, 1, keepdim=True)), 1)

    def heatmap_to_scale(self, heatmap_logits):
        """
        :param heatmap_logits: (batch_size, num_keypoints, self.heatmap_size ** 2)
        :return: (batch_size, num_keypoints)
        """
        scales = torch.mean(heatmap_logits, dim=2)
        scales = scales - torch.max(scales, 1, keepdim=True)[0]
        scales = nn.functional.softmax(scales, dim=1)

        return scales

    def heatmap_to_std(self, heatmap, xy_normalized):
        """
        :param heatmap: (batch_size, heatmap_size, heatmap_size)
        :param xy_normalized: (batch_size, 2)
        :return: (batch_size, 1)
        """
        mesh_grid = torch.stack([self.ranx, self.rany], 2)
        var = torch.sum(torch.sum((mesh_grid[None] - xy_normalized[:, None, None]) ** 2, 3) * heatmap, dim=(1, 2))
        std = torch.sqrt(var)[:, None]

        if DEBUG:
            print('std max {}, min {}, mean {}'.format(std.max().item(), std.min().item(), std.mean().item()))

        return std

    def encode(self, images, rsample=True, u_shift=None, v_shift=None):
        batch_size = images.shape[0]
        images = images.view(batch_size, self.num_cameras * self.frame_stack * self.image_channels,
                             self.image_size, self.image_size)
        latent = self.encoder(images * 2 - 1)
        heatmap_logits = self.heatmap_head(latent) * 5.0
        latent_z_map = self.z_head(latent).view(batch_size * self.num_cameras * self.num_keypoints,
                                                self.heatmap_size, self.heatmap_size)
        z_map = - (torch.sigmoid(latent_z_map) * 2 * self.mean_depth)
        if DEBUG:
            print('heatmap logits max {},mean log z {}'.format(heatmap_logits.max().item(), latent_z_map.mean().item()))
        heatmap = nn.functional.softmax(heatmap_logits.reshape(-1, self.heatmap_size ** 2), dim=1)
        heatmap = heatmap.reshape(-1, self.heatmap_size, self.heatmap_size)
        xy_normalized = self.heatmap_to_xy(heatmap)
        scales = self.heatmap_to_scale(heatmap_logits.reshape(-1, self.num_keypoints, self.heatmap_size ** 2))
        std = self.heatmap_to_std(heatmap, xy_normalized)
        # add noise on camera uv plane
        if self.noise > 0 and rsample:
            xy_normalized = xy_normalized + std * torch.clamp(torch.randn_like(xy_normalized) * self.noise, -1.0, 1.0)
        z = self.compute_z(heatmap, z_map)

        xy_normalized = xy_normalized.reshape(batch_size, self.num_cameras, self.num_keypoints, 2)
        xy_normalized = self.add_offset_xy(xy_normalized, u_shift, v_shift)
        xy_normalized = xy_normalized.reshape(-1, 2)

        xyzw_cam = self.unproject(xy_normalized, z).reshape(batch_size, self.num_cameras, self.num_keypoints, 4)
        xyzw_world = torch.einsum('cnm,bckm->bckn', self.V_invs, xyzw_cam)
        xyz_world = (xyzw_world / (xyzw_world[:, :, :, 3:4] + epsilon))[:, :, :, :3]  # (batch, num_cam, num_keypoints, 3)
        multiview_loss = self.multiview_loss(xyz_world)
        scales = scales.reshape(batch_size, self.num_cameras, self.num_keypoints, 1)
        sparsity_loss = self.sparsity_loss(scales)

        '''
        clip keypoints to prevent NaN
        '''
        xy_world, z_world = torch.split(xyz_world, [2, 1], 3)
        # xy_world = torch.clamp(xy_world, min=-3.0, max=3.0)
        # z_world = torch.clamp(z_world, min=-3.0, max=3.0)

        xyz_world = torch.cat([xy_world, z_world], -1)

        separation_loss = self.separation_loss(xyz_world.mean(1))

        loss_dict = dict(sparsity_loss=sparsity_loss,
                         multiview_loss=multiview_loss,
                         separation_loss=separation_loss
                         )

        heatmap = heatmap.reshape(batch_size, self.num_cameras, self.num_keypoints, self.heatmap_size, self.heatmap_size)

        # weight with confidence
        xyz_world = xyz_world * scales / scales.mean(dim=1, keepdim=True)
        xyzs_world = torch.cat([xyz_world, scales], -1)
        xyzs_world = xyzs_world.mean(1)  # (batch_size, num_keypoints, 4)

        return xyzs_world, loss_dict, heatmap

    def multiview_loss(self, xyz_world):
        """
        :param xyz_world: Tensor (batch, num_cam, num_points, 3)
        :return: multiview_loss: (batch, 1)
        """
        batch_size, _, num_points, _ = xyz_world.shape

        xyz_world_target = xyz_world[:, :, None].expand(-1, -1, self.num_cameras, -1, -1).detach()
        xyz_world = xyz_world[:, None].expand(-1, self.num_cameras, -1, -1, -1)
        multiview_loss = nn.functional.mse_loss(xyz_world, xyz_world_target)

        return multiview_loss

    def separation_loss(self, xyz):
        """
        :param xyz: (batch, num_points, 3)
        :return: loss
        """
        _, num_points, _ = xyz.shape
        xyz0 = xyz[:, :, None, :].expand(-1, -1, num_points, -1)
        xyz1 = xyz[:, None, :, :].expand(-1, num_points, -1, -1)
        sq_dist = torch.sum((xyz0 - xyz1) ** 2, -1)
        loss = 1 / (1000 * sq_dist + 1)
        return torch.mean(loss, [1, 2])

    def sparsity_loss(self, scales):
        loss = torch.sum(torch.abs(scales), (1, 2), keepdim=True)
        # disable sparsity loss since sum to 1 anyway
        return torch.zeros_like(loss)

    def keypoints_to_heatmap(self, keypoints, u_shift, v_shift):
        """
        :param keypoints: Tensor size(batch, num_keypoints, 4)
               joints_xyz: Tensor size(batch, num_joints, 3)
               joints: Tensor size(batch, num_joints)
        :return: images: Tensor size(batch, num_cam, num_keypoints, h, w)
        """
        xyz_world, scale = torch.split(keypoints, [3, 1], 2)
        # if self.noise > 0:
        #     xyz_world = xyz_world + (torch.rand_like(xyz_world) * self.noise * 2 - self.noise)
        keypoints_heatmap = self.xyz_world_to_heatmap(xyz_world, u_shift, v_shift, scale)

        return keypoints_heatmap

    def visualize(self, images, keypoints, images_hat, heatmap, u_shift, v_shift):
        """
        :param images: (batch_size, num_cameras, 3, image_h, image_w)
        :param keypoints:  (batch_size, num_keypoints, 4)
        :param images_hat:  (batch_size, num_cameras, 3, image_h, image_w)
        :param heatmap:  (batch_size, num_cameras, num_keypoints, heatmap_size, heatmap_size)
        :return:
        """
        images_hat = torch.clamp(images_hat, 0.0, 1.0)
        batch_size, num_cameras, _, image_h, image_w = images.shape
        _, num_keypoints, _ = keypoints.shape
        xyz_world, scale = keypoints.split([3, 1], dim=2)
        xy_normalized = self.xyz_world_to_xy_normalized(xyz_world)

        att_threshold = 1.0 / num_keypoints * 0.0

        keypoints_heatmap = self.keypoints_to_heatmap(keypoints, u_shift, v_shift)

        keypoints_heatmap = keypoints_heatmap.reshape(batch_size * num_cameras, num_keypoints * self.heatmap_size * self.heatmap_size)
        keypoints_heatmap = keypoints_heatmap / keypoints_heatmap.max(dim=1, keepdim=True)[0]
        keypoints_heatmap = self.upsampler(keypoints_heatmap.view(batch_size * num_cameras * num_keypoints, 1, self.heatmap_size, self.heatmap_size))
        keypoints_heatmap = keypoints_heatmap.view(batch_size, num_cameras, num_keypoints, 1, image_h, image_w)
        keypoints_heatmap = keypoints_heatmap.repeat(1, 1, 1, 3, 1, 1)


        heatmap = heatmap.reshape(batch_size * num_cameras * num_keypoints, self.heatmap_size * self.heatmap_size)
        heatmap = heatmap / heatmap.max(dim=1, keepdim=True)[0]
        heatmap = self.upsampler(heatmap.view(batch_size * num_cameras * num_keypoints, 1,
                                              self.heatmap_size, self.heatmap_size))
        heatmap = heatmap.view(batch_size, num_cameras, num_keypoints, 1, image_h, image_w)
        heatmap = heatmap.repeat(1, 1, 1, 3, 1, 1)

        """
        images (batch_size, num_cameras, 3, image_h, image_w)
        keypoints (batch_size, num_keypoints, 3)
        heatmap (batch_size, num_cameras, num_keypoints, 3, image_h, image_w)
        images_hat (batch_size, num_cameras, 3, image_h, image_w)
        xy_normalized (batch_size, num_cameras, num_keypoints, 2)
        """

        images = (images.permute(0, 1, 3, 4, 2).detach() * 255.0).to('cpu', torch.uint8).numpy()
        images_hat = (images_hat.permute(0, 1, 3, 4, 2).detach() * 255.0).to('cpu', torch.uint8).numpy()
        keypoints_heatmap = (keypoints_heatmap.permute(0, 1, 2, 4, 5, 3).detach() * 255.0).to('cpu', torch.uint8).numpy()
        heatmap = (heatmap.permute(0, 1, 2, 4, 5, 3).detach() * 255.0).to('cpu', torch.uint8).numpy()
        xy_normalized = xy_normalized.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()

        canvases = []
        for b in range(batch_size):
            canvas = []
            for c1 in range(num_cameras):
                images_vis = images[b, c1].copy()
                xy_vis = images[b, c1].copy()

                for i, (x, y) in enumerate(xy_normalized[b, c1]):
                    x = x / 2 + 0.5
                    y = y / 2 + 0.5
                    if scale[b, i] > att_threshold:
                        cv2.circle(xy_vis, (int(x * image_w), int((1 - y) * image_h)), 1,
                                   self.cmap[i].tolist(), 2, lineType=-1)

                heatmap_vis = []
                for i, (x, y) in enumerate(xy_normalized[b, c1]):
                    x = x / 2 + 0.5
                    y = y / 2 + 0.5
                    heatmap_vis.append(heatmap[b, c1, i].copy())
                    cv2.circle(heatmap_vis[-1], (int(x * image_w), int((1 - y) * image_h)),
                               2, self.cmap[i].tolist(), 2, lineType=-1)
                heatmap_vis = np.stack(heatmap_vis, 0)
                heatmap_vis = heatmap_vis.transpose(1, 0, 2, 3).reshape(image_h, num_keypoints * image_w, 3)

                gaussian_vis = []
                for i, (x, y) in enumerate(xy_normalized[b, c1]):
                    x = x / 2 + 0.5
                    y = y / 2 + 0.5
                    gaussian_vis.append(keypoints_heatmap[b, c1, i].copy())
                    # cv2.circle(gaussian_vis[-1], (int(x * image_w), int((1 - y) * image_h)),
                    #            2, self.cmap[i].tolist(), 2, lineType=-1)
                gaussian_vis = np.stack(gaussian_vis, 0)
                gaussian_vis = gaussian_vis.transpose(1, 0, 2, 3).reshape(image_h, num_keypoints * image_w, 3)

                images_hat_vis = images_hat[b, c1]

                canvas.append(np.concatenate([images_vis, xy_vis, images_hat_vis, heatmap_vis, gaussian_vis], 1))
            canvases.append(np.concatenate(canvas, 0))

        canvases = np.stack(canvases)

        return torch.from_numpy(canvases / 255.0).permute(0, 3, 1, 2)
