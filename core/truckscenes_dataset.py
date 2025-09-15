
import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from truckscenes import TruckScenes
from core.options import Options

from core.trucksc import fused_radar_pc, trucksc_read, frame_conversion

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



CAMERA_VIEW_NAMES = ['CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_BACK', 'CAMERA_LEFT_BACK']


class SquarePadPIL:
    def __call__(self, image: Image.Image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        pad_left = hp
        pad_right = max_wh - w - hp
        pad_top = vp
        pad_bottom = max_wh - h - vp
        padding = (pad_left, pad_top, pad_right, pad_bottom)  # (left, top, right, bottom)
        return TF.pad(image, padding, fill=0, padding_mode='constant')


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

# Concat plucker embedding directly to the latent tensor
# https://github.com/gohyojun15/SplatFlow/blob/main/model/gsdecoder/camera_embedding.py#L26
# Plucker embedding from Nuscenes dataset
# https://github.com/valeoai/LaRa/blob/830230ab8dfd67eabe74dd92e03d4881cd950d60/semanticbev/datamodules/components/nuscenes_data.py#L170
# https://github.com/valeoai/LaRa/blob/830230ab8dfd67eabe74dd92e03d4881cd950d60/semanticbev/models/components/LaRa_embeddings.py#L51
# SEVA - Stable Virtual Camera
# https://github.com/Stability-AI/stable-virtual-camera/blob/fe19948e9b7bea261ab2db780a59656131404a83/seva/geometry.py#L119
def create_plucker_embedding(sensor_sd_token, ref_sd_token, trucksc: TruckScenes, device: torch.device, dtype: torch.dtype, hw=512):
    _, _, transformation_matrix = frame_conversion.sensor_from_sensor(trucksc, sensor_sd_token, ref_sd_token)

    sensor_sd_rec = trucksc.get('sample_data', sensor_sd_token)
    calibrated_sensor_t = trucksc.get('calibrated_sensor', sensor_sd_rec['calibrated_sensor_token'])
    intrinsics = calibrated_sensor_t['camera_intrinsic']

    transformation_matrix = torch.tensor(transformation_matrix, device=device, dtype=dtype)
    transformation_matrix = transformation_matrix[:3, :]  # Get non-homogeneous 3d points later
    intrinsics = torch.tensor(intrinsics, device=device, dtype=dtype)
    
    # We resize and pad the image, then we encode the image to a latent.
    updated_intrinsics = intrinsics.clone().to(device=device, dtype=torch.float32)
    # Add the amount of padding on top of the image
    padding_top = int((sensor_sd_rec['width']-sensor_sd_rec['height'])/2)
    updated_intrinsics[1, 2] += padding_top
    # Divide by stride to get resized intrinsics, hw is the resolution of the image
    resize_factor = sensor_sd_rec['width']/hw
    updated_intrinsics[0, 0] /= resize_factor
    updated_intrinsics[0, 2] /= resize_factor
    updated_intrinsics[1, 1] /= resize_factor
    updated_intrinsics[1, 2] /= resize_factor

    y_range = torch.arange(hw, device=device, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(hw, device=device, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    xy_grid_hom = to_hom(xy_grid)

    # Camera rays at camera coordinate system
    with torch.amp.autocast("cuda"):
        grid_3d_cam = xy_grid_hom @ updated_intrinsics.inverse().transpose(-1, -2)
        # grid_3d_cam = xy_grid_hom @ intrinsics.to(dtype=torch.float32).inverse().transpose(-1, -2)
    grid_3d_cam = grid_3d_cam.to(dtype=dtype)

    # Camera rays at reference coordinate system
    grid_3d_cam_hom = to_hom(grid_3d_cam)
    grid_3d_ref = grid_3d_cam_hom @ transformation_matrix.transpose(-1, -2)

    center_3D_cam = torch.zeros_like(grid_3d_cam)
    center_3D_cam_hom = to_hom(center_3D_cam)
    center_3D_ref = center_3D_cam_hom @ transformation_matrix.transpose(-1, -2)
    rays = grid_3d_ref - center_3D_ref

    # Unit vectors of camera rays
    rays = torch.nn.functional.normalize(rays, dim=-1)

    camera_origins = center_3D_ref  # Equal to translation_matrix = transformation_matrix[:3, 3]
    moments = torch.cross(camera_origins, rays, dim=-1)
    
    # plucker = torch.cat((camera_origins, rays, moments), dim=-1)
    plucker = torch.cat((rays, moments), dim=-1)
    plucker = plucker.permute(1, 0).reshape(-1, hw, hw)
    return plucker


class MANTruckscenesDataset(Dataset):
    def __init__(
        self,
        opt: Options,
        data_dir="/data/man-truckscenes-mini",
        eval_split="mini_train",
        trucksc_version="v1.0-mini",
        training=True,
        **kwargs,
    ):
        self.training = training
        self.data_dir = data_dir[0] if isinstance(data_dir, list) else data_dir
        self.opt = opt
        self.trucksc = TruckScenes(trucksc_version, self.data_dir, False)
        self.trucksc_dataroot = self.trucksc.dataroot
        scene_tokens = trucksc_read.split_scene_tokens(self.trucksc, eval_split, illuminated_scenes_only=True)

        self.all_frames_tokens = []
        for scene_token in scene_tokens:
            scene_samples_data = trucksc_read.get_scene_samples_data(self.trucksc, scene_token)
            for sample_data in scene_samples_data:
                self.all_frames_tokens.append(sample_data)

        self.camera_token_to_path = {}
        for sample_data in self.all_frames_tokens:
            for camera_view_name in CAMERA_VIEW_NAMES:
                camera_token = sample_data[camera_view_name]
                camera_data = self.trucksc.get('sample_data', camera_token)
                self.camera_token_to_path[camera_token] = osp.join(self.trucksc_dataroot, camera_data['filename'])
        
        self.dataset = self.all_frames_tokens[:1]
        self.ori_imgs_nums = len(self)
        print(f"Dataset samples: {len(self.dataset)}")


    def getdata(self, idx):
        sample_data = self.dataset[idx]
        ref_sensor = 'LIDAR_LEFT'
        ref_token = sample_data[ref_sensor]
         
        sd_rec = self.trucksc.get('sample_data', ref_token)
        sample_rec = self.trucksc.get('sample', sd_rec['sample_token'])  

        # Load camera images at time t
        camera_images_t = []
        for camera_view_name in CAMERA_VIEW_NAMES:
            camera_view_token_t = sample_data[camera_view_name]
            camera_image_path_t = self.camera_token_to_path[camera_view_token_t]

            camera_image_t = Image.open(camera_image_path_t).convert("RGB")

            # 直接缩放到 input_size × input_size（会拉伸）
            camera_image_t = T.Resize((self.opt.input_size, self.opt.input_size))(camera_image_t)

            camera_image_t = torch.from_numpy(np.array(camera_image_t).astype(np.float32) / 255.0)  # [H,W,3]
            camera_images_t.append(camera_image_t.permute(2, 0, 1).contiguous())

        camera_tensor_t = torch.stack(camera_images_t, dim=0)
        
        # Create plucker embedding
        plucker_embeddings_t = []
        for camera_view_name in CAMERA_VIEW_NAMES:
            camera_token_t = sample_data[camera_view_name]
            plucker_embedding_t = create_plucker_embedding(camera_token_t, ref_token, self.trucksc, camera_tensor_t.device, camera_tensor_t.dtype, hw=self.opt.input_size)
            plucker_embeddings_t.append(plucker_embedding_t)
        plucker_embedding_t = torch.stack(plucker_embeddings_t, dim=0)

        images_input = TF.normalize(camera_tensor_t, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        images_input = torch.cat([images_input, plucker_embedding_t], dim=1) # [V=4, 9, H, W]

        # point_cloud_tensor = fused_radar_pc.get_fused_radar_pointcloud(self.trucksc, sample_data)
        # point_cloud_tensor = fused_radar_pc.filter_fused_radar_pointcloud(point_cloud_tensor, self.trucksc, sample_rec, set_z=1.0)

        pcl = fused_radar_pc.get_fused_radar_pointcloud(
            self.trucksc, sample_rec
        )
        pcl = fused_radar_pc.filter_fused_radar_pointcloud(
            pcl, self.trucksc, sample_rec, set_z=1.0
        )
        point_cloud = pcl.points
        K = 1296    # controlled by the input size and patch size
        P = point_cloud.shape[1]
        replace = P < K
        idxs = np.random.choice(P, size=K, replace=replace)
        pts  = point_cloud[0:3, idxs].T.astype(np.float32)        # [K,3]
        vrel = point_cloud[3:6, idxs].T.astype(np.float32)        # [K,3]
        rcs  = point_cloud[6, idxs][:, None].astype(np.float32)   # [K,1]
        dop  = point_cloud[7, idxs][:, None].astype(np.float32)   # [K,1]

        
        results = {}
        results['input'] = {'images': images_input}
        results['radar_gt'] = {
            'points': torch.from_numpy(pts),   # [P,3] -> pcd.pc_data['x/y/z']
            'rcs':    torch.from_numpy(rcs),      # [P,1] -> pcd.pc_data['rcs']
            'vrel':   torch.from_numpy(vrel),     # [P,3] -> pcd.pc_data['vrel_x/y/z']
            'doppler': torch.from_numpy(dop)
        }
        return results

        # return (
        #     images_input,                        # 0
        #     pcl,                                 # 1
        # )
    
    def __getitem__(self, idx):
        data = self.getdata(idx)
        return data

    def __len__(self):
        return len(self.dataset)
