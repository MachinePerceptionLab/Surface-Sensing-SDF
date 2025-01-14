import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 num_views=-1,  
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        # used a fake depth image and normal image
        self.depth_images = []
        self.normal_images = []

        for path in image_paths:
            depth = np.ones_like(rgb[:, :1])
            self.depth_images.append(torch.from_numpy(depth).float())
            normal = np.ones_like(rgb)
            self.normal_images.append(torch.from_numpy(normal).float())
            
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
            
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx],
        }
        
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["mask"] = torch.ones_like(self.depth_images[idx][self.sampling_idx, :])
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']


# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        gt_depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "depth/*.npy"))
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.depth_intrinsic = []
        self.depth_pose = []
        for world_mat in world_mats:
            P = world_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.depth_intrinsic.append(torch.from_numpy(intrinsics).float())
            self.depth_pose.append(torch.from_numpy(pose).float())

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.scale_mat = torch.from_numpy(scale_mat).float()

        self.rgb_images = []
        self.rgb_gt = []
        self.grey_gt = []
        self.grey_images = []
        for path in image_paths:
            grey = torch.from_numpy(cv2.imread(path,cv2.IMREAD_GRAYSCALE)).float()/ 255.0
            self.grey_gt.append(grey)
            self.grey_images.append(grey.reshape(1, -1).transpose(1, 0))
            rgb = rend_util.load_rgb(path)
            self.rgb_gt.append(torch.from_numpy(rgb).float())
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []
        self.depth_gt = []
        self.gt_depth_images = []
        self.gt_depth_images_warp = []
        self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F)

        for dpath, npath,gtdpath in zip(depth_paths, normal_paths,gt_depth_paths):
            depth = np.load(dpath)
            self.depth_gt.append(torch.from_numpy(depth).float())
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())

            depth = np.load(gtdpath).astype(np.float32)
            depth_filtered = self.depth_cleaner.apply(depth)
            depth_filtered[depth == 0.] = 0.
            depth_filtered = np.nan_to_num(depth_filtered)
            self.gt_depth_images_warp.append(torch.from_numpy(depth_filtered).float())
            self.gt_depth_images.append(torch.from_numpy(depth_filtered.reshape(-1, 1)).float())

            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        idx_src = [(idx-4+self.n_images)%self.n_images,
         (idx-3+self.n_images)%self.n_images,
         (idx-2+self.n_images)%self.n_images,
         (idx-1+self.n_images)%self.n_images,
         (idx+1+self.n_images)%self.n_images,
         (idx+2+self.n_images)%self.n_images,
         (idx+3+self.n_images)%self.n_images,
         (idx+4+self.n_images)%self.n_images,]

        # idx_src = [(idx+self.n_images)%self.n_images,]
        
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "intrinsics_src": torch.stack([self.intrinsics_all[i] for i in idx_src],dim=0),
            "pose_src":  torch.stack([self.pose_all[i] for i in idx_src],dim=0),
            "depth_intrinsic": self.depth_intrinsic[idx],
            "depth_pose": self.depth_pose[idx],
            "scale_mat": self.scale_mat,
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx],
            "grey": self.grey_gt[idx],
            "warp_depth": self.depth_gt[idx],
            "src_grey": torch.stack([self.grey_gt[i] for i in idx_src],dim=0),
            "gt_depth": self.gt_depth_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_gt[idx]                
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["warp_depth"] = self.depth_gt[idx]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
            ground_truth["grey"] = self.grey_gt[idx]
            ground_truth["src_grey"] = torch.stack([self.grey_gt[i] for i in idx_src],dim=0)

            ground_truth["gt_depth"] = self.gt_depth_images[idx][self.sampling_idx, :]
            ground_truth["warp_gt_depth"] = self.gt_depth_images_warp[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']