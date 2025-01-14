import torch
from torch import nn
import utils.general as utils
import math
import utils.rend_util as rend_util
import torch.nn.functional as F
from math import exp, sqrt
import numpy as np
from scipy.spatial import cKDTree
import trimesh

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None
  
    def forward(self, prediction, target, mask):

        # scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = prediction

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    
    
class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 end_step = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        return self.depth_loss(depth_pred.reshape(1, 32, 32), depth_gt.reshape(1, 32, 32),
         mask.reshape(1, 32, 32))
        
    def get_normal_loss(self, normal_pred, normal_gt):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        return l1, cos

    def update_patch_size(self, h_patch_size, device):
        offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
        return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 2
    def sample_near_points(self,pointcloud,device = 'cpu'):
        POINT_NUM_GT = pointcloud.shape[0] 
        QUERY_EACH = 10000//POINT_NUM_GT

        ptree = cKDTree(pointcloud)
        sigmas = []
        d = ptree.query(pointcloud,51)
        sigmas.append(d[0][:,-1])    
        sigmas = np.concatenate(sigmas)
        sample = []
        for i in range(QUERY_EACH):
            scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
            tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
            sample.append(tt)
        sample = np.asarray(sample).reshape(-1,3)
        sample = torch.from_numpy(sample).to(device).float()
        return sample,QUERY_EACH

    def sample_near_points_cone(self,pointcloud,interval, device = 'cpu'):
        POINT_NUM_GT = pointcloud.shape[0] 
        QUERY_EACH = 10000//POINT_NUM_GT

        sample = []
        for i in range(QUERY_EACH):
            tt = pointcloud + interval[:,None] *  torch.normal(0., 1., size=pointcloud.shape,device="cuda")
            sample.append(tt)
        sample = torch.stack(sample).reshape(-1,3).to(device).float()
        return sample,QUERY_EACH
        
    def compute_LNCC(self, ref_gray, src_grays,warping_mask):

        # ref_gray: [1, 121, bs, 1]
        # src_grays: [nsrc, 121, bs, 1]
        ref_gray = ref_gray.permute(2, 0, 3, 1)  # [batch_size, 1, 3, 121]
        src_grays = src_grays.permute(2, 0, 3, 1)  # [batch_size, nsrc, 3, 121]

        ref_src = ref_gray * src_grays  # [batch_size, nsrc, 1, npatch]

        bs, nsrc, nc, npatch = src_grays.shape
        patch_size = int(sqrt(npatch))
        ref_gray = ref_gray.view(bs, 1, 1, patch_size, patch_size).view(-1, 1, patch_size, patch_size)
        src_grays = src_grays.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)
        ref_src = ref_src.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)

        ref_sq = ref_gray.pow(2)
        src_sq = src_grays.pow(2)

        filters = torch.ones(1, 1, patch_size, patch_size, device=ref_gray.device)
        padding = patch_size // 2

        ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)

        u_ref = ref_sum / npatch
        u_src = src_sum / npatch

        cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
        ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
        src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

        cc = cross * cross / (ref_var * src_var + 1e-10)  # [batch_size, nsrc]
        ncc = 1 - cc
        aa = torch.clamp(ncc, 0.0, 2.0)
        ncc, mask_idx = torch.topk(aa, 3, dim=1, largest=False)
        top4_mask = torch.gather(warping_mask, 1, mask_idx).float()
        num = torch.sum(top4_mask * ncc, dim=-1) 
        denom = torch.sum(top4_mask, dim=-1)

        valids = denom > 1e-4
        res = num[valids] / denom[valids]
        # ncc = torch.mean(ncc, dim=1, keepdim=True)
        return res

    def depth_confidence(self,model,depth_pred,ground_truth,model_input,mask):
        # depth_pred: [1024, 1]
        # depth_gt: [1,1024, 1]
        # mask: [1,1024, 1] 

        W,H = [384, 384] #### scannet load from conf fataset
        intrinsics = model_input["intrinsics"] #  [1,4,4]
        uv = model_input["uv"] #  [1,1024,2]
        pose = model_input["pose"] #  [1,4,4]

        intrinsics_src = model_input["intrinsics_src"].cuda() #  [1,9,4,4]
        pose_src = model_input["pose_src"].cuda() #  [1,9,4,4]
        src_full_rgb = ground_truth["src_grey"].cuda() # [1,9,384,384]

        depth_intrinsic = model_input["depth_intrinsic"] #  [1,4,4]
        depth_pose = model_input["depth_pose"] #  [1,4,4]
        scale_mat = model_input['scale_mat'][0]

        gt_depth_gt = ground_truth['gt_depth']

        batch_size = uv.shape[1]

        surface_points = rend_util.get_surface_point(uv, depth_pose, depth_intrinsic,gt_depth_gt.squeeze()[None]).reshape(-1, 3) #[1024,3]
        surface_points = (torch.linalg.inv(scale_mat[:3, :3]) @ (surface_points.T - scale_mat[:3, 3:])).T
        surface_points_near = rend_util.get_surface_point(uv-torch.tensor([2,0],device='cuda'), depth_pose, depth_intrinsic,gt_depth_gt.squeeze()[None]).reshape(-1, 3) #[1024,3]
        surface_points_near = (torch.linalg.inv(scale_mat[:3, :3]) @ (surface_points_near.T - scale_mat[:3, 3:])).T

        pixel_world_invertal = torch.linalg.norm((surface_points.detach() - surface_points_near.detach()), ord=2, dim=-1)
        samples,bs = self.sample_near_points_cone(surface_points.detach(),pixel_world_invertal,device = surface_points.device)
        # samples,bs = self.sample_near_points(surface_points.detach().cpu().numpy(),device = surface_points.device)
        samples.requires_grad = True
        sdf_sample,_ ,gradients_sample= model.implicit_network.get_outputs(samples) # 5000x1  # 5000x3
        grad_norm = torch.nn.functional.normalize(gradients_sample, p=2, dim=1) # 5000x3
        sample_moved = samples - grad_norm * sdf_sample # 5000x3

        ## 1024 normal to world coord
        surface_norm = ground_truth['normal'].cuda()[0] @ pose[0,:3,:3].permute(1,0) #  [1024,3]
        surface_norm = torch.nn.functional.normalize(surface_norm, p=2, dim=-1)

        moved_gradient = model.implicit_network.gradient(sample_moved) #[5000,3]
        moved_gradient = torch.nn.functional.normalize(moved_gradient, p=2, dim=-1).detach()
        sample2surface_weight = (torch.sum(moved_gradient.view(bs,-1,3) * surface_norm[None], dim = -1) +1.0)/2. #[bs,1024]

        surface_loss = torch.abs(torch.sum((surface_points[None] - sample_moved.view(bs,-1,3)) * surface_norm, dim = -1)) * sample2surface_weight #[bs,1024]]
        # surface_loss = surface_loss.mean()

        # project and filter
        vertices = torch.cat((sample_moved, torch.ones_like(sample_moved[:, :1])), dim=-1)
        vertices = vertices.permute(1, 0)
        vertices = vertices.float()
    
        # # transform and project src
        w2c = torch.inverse(pose_src).cuda()
        cam_points = intrinsics_src[0] @ w2c[0] @ vertices
        pix_coords = cam_points[:,:2, :] / (cam_points[:,2, :].unsqueeze(1) + 1e-10)
        pix_coords = pix_coords.permute(0,2,1) #[9,1024,2]
        pix_coords[..., 0] /= W - 1
        pix_coords[..., 1] /= H - 1
        pix_coords = (pix_coords - 0.5) * 2 #[9,1024,2]

        in_front = cam_points.permute(0,2,1)[..., 2] > 0 # [9,5000]
        warping_mask = (in_front & (pix_coords < 1).all(dim=-1) & (pix_coords > -1).all(dim=-1)).detach().view(pose_src.shape[1],bs,-1) # [9,bs,1024]
        warping_mask = warping_mask.float().sum(1) > 7. # # [9,1024]

        src_rgb = F.grid_sample(src_full_rgb.permute(1,0,2,3), pix_coords[:, None],
            align_corners=True)[:, :, 0,:].permute(0,2,1).view(pose_src.shape[1],bs,batch_size,1) #[9,121,1024,3]

        # src_rgb = F.grid_sample(src_full_rgb.permute(1,0,2,3), pix_coords[:, None],
        #     mode='nearest',align_corners=True)[:, :, 0,:].permute(0,2,1).view(-1,batch_size,total_size,1) #[2,1024,3]
        # target_rgb = F.grid_sample(ground_truth["grey"][None].cuda(), (2 * uv[None] / (W - 1) - 1.0),
        #     align_corners=True)[0:, :, 0,:].permute(0,2,1).view(-1,batch_size,total_size,1) #[1024,3]

        # # transform and project target
        w2c_tar = torch.inverse(pose).cuda()
        cam_points_tar = intrinsics[0] @ w2c_tar[0] @ vertices
        pix_coords_tar = cam_points_tar[:2, :] / (cam_points_tar[2, :].unsqueeze(0) + 1e-10)
        pix_coords_tar = pix_coords_tar.permute(1,0) #[5000,2]
        uv_tar = pix_coords_tar.detach().clone()
        pix_coords_tar[..., 0] /= W - 1
        pix_coords_tar[..., 1] /= H - 1
        pix_coords_tar = (pix_coords_tar - 0.5) * 2 #[5000,2]
        target_rgb = F.grid_sample(ground_truth["grey"][None].cuda(), pix_coords_tar[None, None],
            align_corners=True)[:, :, 0,:].permute(0,2,1).view(1,bs,batch_size,1) #[1,121,1024,3]
        # sampled_mask = sampled_mask + (1. - valid)

        ncc = self.compute_LNCC(target_rgb,src_rgb,warping_mask.permute(1,0)) # the prob of similarity, 1 is the same [1024,1]
        ncc = 0.5 * (ncc.sum(dim=0) / ncc.shape[0] + 1e-10).squeeze(-1)

        # wrap depth from ref depth map
        cc = ground_truth["warp_gt_depth"].cuda()
        target_depth = F.grid_sample(cc[None], pix_coords_tar[None, None],align_corners=True).squeeze().detach() #[1024]
        gt_moved = rend_util.get_surface_point(uv_tar[None], depth_pose, depth_intrinsic,target_depth[None]).reshape(-1, 3) #[1024,3]
        gt_moved = (torch.linalg.inv(scale_mat[:3, :3]) @ (gt_moved.T - scale_mat[:3, 3:])).T
        valid = (target_depth > 0.0) & (torch.linalg.norm((gt_moved - sample_moved), ord=2, dim=-1)  < 0.015 ).detach()
        # manhattan distance
        sdf_loss = torch.sum(torch.abs(gt_moved[valid] - sample_moved[valid]), dim = -1).mean()
        surface_loss = surface_loss.view(-1)[valid].mean()

        return surface_loss, ncc   
          
    def sdf_loss(self,model,depth_pred,ground_truth,model_input,mask):
        # depth_pred: [1024, 1]
        # depth_gt: [1,1024, 1]
        # mask: [1,1024, 1] 

        W,H = [384, 384] #### scannet load from conf fataset
        intrinsics = model_input["intrinsics"] #  [1,4,4]
        uv = model_input["uv"] #  [1,1024,2]
        pose = model_input["pose"] #  [1,4,4]

        intrinsics_src = model_input["intrinsics_src"].cuda() #  [1,9,4,4]
        pose_src = model_input["pose_src"].cuda() #  [1,9,4,4]
        src_full_rgb = ground_truth["src_grey"].cuda() # [1,9,384,384]

        depth_intrinsic = model_input["depth_intrinsic"] #  [1,4,4]
        depth_pose = model_input["depth_pose"] #  [1,4,4]
        scale_mat = model_input['scale_mat'][0]

        gt_depth_gt = ground_truth['gt_depth']

        batch_size = uv.shape[1]
        depth_valid = gt_depth_gt != 0 #[1,1024, 1]
        depth_valid = depth_valid.squeeze()
        uv = uv[:,depth_valid,:]
        gt_depth_gt = gt_depth_gt[:,depth_valid,:]

        surface_points = rend_util.get_surface_point(uv, depth_pose, depth_intrinsic,gt_depth_gt.squeeze()[None]).reshape(-1, 3) #[1024,3]
        surface_points = (torch.linalg.inv(scale_mat[:3, :3]) @ (surface_points.T - scale_mat[:3, 3:])).T
        surface_points.requires_grad = True
        pts2sdf = model.implicit_network.get_sdf_vals(surface_points)
        sdf_loss = F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf),
                                reduction='sum') / pts2sdf.shape[0]
        return sdf_loss     

    def forward(self, model,model_outputs, ground_truth,model_input):
        rgb_gt = ground_truth['rgb'].cuda()
        # monocular depth and normal
        # depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()

        model_input["depth_intrinsic"] = model_input["depth_intrinsic"].cuda() #  [1,4,4]
        model_input["depth_pose"] = model_input["depth_pose"].cuda() #  [1,4,4]
        model_input['scale_mat'] = model_input['scale_mat'].cuda()
        ground_truth['gt_depth'] = ground_truth['gt_depth'].cuda()

        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]

        if len(model_outputs) ==3 :
            mask = (ground_truth['mask'] > 0.5).cuda()
            return {'ncc_loss': None}
            # ncc = self.depth_confidence(model,depth_pred.cuda(),depth_gt,ground_truth,model_input,mask)
            # return  {'ncc_loss': ncc[None]}
            # return {'ncc_loss': None}

        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        sdf_loss = self.sdf_loss(model,depth_pred,ground_truth,model_input,mask)
        surface_loss,ncc = self.depth_confidence(model,depth_pred,ground_truth,model_input,mask)
        # ncc_loss = 0.5 * (ncc.sum(dim=0) / ncc.shape[0]).squeeze(-1)

        depth_gt, depth_mask = rend_util.convert_gtdepth(model_input,ground_truth)

        depth_loss = self.get_depth_loss(depth_pred, depth_gt, depth_mask)
        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt)
        
        smooth_loss = self.get_smooth_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.smooth_weight * smooth_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.normal_l1_weight *  normal_l1 +\
               decay * self.normal_cos_weight * normal_cos+\
                sdf_loss  + surface_loss + ncc


        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos,
            'ncc_loss': ncc,
            'sdf_loss':sdf_loss,
            'surface_loss': surface_loss,
        }

        return output
