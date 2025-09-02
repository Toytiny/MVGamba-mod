
import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from truckscenes import TruckScenes

from core.trucksc import fused_radar_pc, trucksc_read

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



CAMERA_VIEW_NAMES = ['CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_BACK', 'CAMERA_LEFT_BACK']



class MANTruckscenesDataset(Dataset):
    def __init__(
        self,
        data_dir="/data/man-truckscenes-mini",
        eval_split="mini_val",
        trucksc_version="v1.0-mini",
        transform=None,
        resolution=256,
        max_length=300,
        img_extension=".png",
        **kwargs,
    ):
        self.data_dir = data_dir[0] if isinstance(data_dir, list) else data_dir
        self.transform = transform
        self.resolution = resolution
        self.max_length = max_length
        self.default_prompt = "prompt"
        self.img_extension = img_extension

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
        
        # del self.trucksc

        self.dataset = self.all_frames_tokens
        self.ori_imgs_nums = len(self)
        print(f"Dataset samples: {len(self.dataset)}")
        print(f"Text max token length: {self.max_length}")


    def getdata(self, idx):
        sample_data = self.dataset[idx]
        ref_sensor = 'LIDAR_LEFT'
        ref_token = sample_data[ref_sensor]


        # Load camera images at time t
        camera_images_t = []
        for camera_view_name in CAMERA_VIEW_NAMES:
            camera_view_token_t = sample_data[camera_view_name]
            camera_image_path_t = self.camera_token_to_path[camera_view_token_t]
            camera_image_t = Image.open(camera_image_path_t)
            camera_image_t = torch.from_numpy(np.array(camera_image_t).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
            camera_images_t.append(camera_image_t)
        camera_tensor_t = torch.stack(camera_images_t, dim=0)

        # Create plucker embedding
        # plucker_embeddings_t = []
        # for camera_view_name in CAMERA_VIEW_NAMES:
        #     camera_token_t = sample_data[camera_view_name]
        #     plucker_embedding_t = create_plucker_embedding(camera_token_t, ref_token, self.trucksc, camera_tensor_t.device, camera_tensor_t.dtype)
        #     plucker_embeddings_t.append(plucker_embedding_t)
        # plucker_embedding_t = torch.stack(plucker_embeddings_t, dim=0)

        # if self.plucker_ray:
        #     # build ray embeddings for input views
        #     rays_embeddings = []
        #     for i in range(self.opt.num_input_views):
        #         rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
        #         rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        #         rays_embeddings.append(rays_plucker)
        #     rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        #     images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        #     images_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        #     final_input = {'images': images_input, 'camposes': cam_poses_input} # cam_pose embeded as plucker_rays,
        # else:
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        # final_input = {'images': images_input} #, 'camposes': cam_poses_input}

        point_cloud_tensor = fused_radar_pc.get_fused_radar_pointcloud(self.trucksc, sample_data)
        point_cloud_tensor = fused_radar_pc.filter_fused_radar_pointcloud(point_cloud_tensor, self.trucksc, sample_data, set_z=1.0)

        return (
            images_input,                     # 0
            point_cloud_tensor,                  # 1
            # plucker_embedding_t,                 # 2
        )
    
    # Get error instantly. Can be used for debugging
    def __getitem__(self, idx):
        data = self.getdata(idx)
        return data

    # def __getitem__(self, idx):
    #     for _ in range(10):
    #         try:
    #             data = self.getdata(idx)
    #             return data
    #         except Exception as e:
    #             print(f"Error details: {str(e)}")
    #             idx = idx + 1
    #     raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

