import torch
import torch.nn as nn

from model.pointpillars import *
from model.blocks import *


class TerrainTraversabilityEncoder(nn.Module):

    """
    Model that predicts angle-conditioned traversability distributions, by
    explicitly encoding the approach angle.
    """

    def __init__(self, config):

        super().__init__()

        self.config = config

        # Construct Point Pillars Feature Extractor
        self.pillar_layer = PillarLayer(
            voxel_size = config['voxel_size'],
            point_cloud_range = config['point_cloud_range'], 
            max_num_points = config['max_num_points'], 
            max_voxels = config['max_voxels']
        )
        self.pillar_encoder = PillarEncoder(
            voxel_size = config['voxel_size'], 
            point_cloud_range = config['point_cloud_range'], 
            in_channel=8, 
            out_channel=config['pointpillar_out_channel']
        )

        self.num_bins = config['num_bins']

        # Backbone to encode the BEV feature map
        self.pcd_backbone = build_conv(config = config['pcd_backbone'])

        # Head to encode the heading/approach angle
        self.heading_backbone = build_head(config = config['heading_backbone'])

        # Final MLP to predict the traversability distribution
        self.head = build_head(config = config['head'])

        # Sigmoid activation to produce valid concentration parameters
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        batched_pts = input_dict['batched_pts']

        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        
        # B x pcd_feature_dim x 1 x 1
        pcd_features = self.pcd_backbone(pillar_features).squeeze(-1).squeeze(-1)

        heading_angles = input_dict['heading_angles']

        # # Repeat the point cloud features to match the heading angle dimension
        pcd_features = pcd_features.unsqueeze(1).repeat(1, heading_angles.shape[1], 1)
        pcd_features = pcd_features.view(-1, pcd_features.shape[-1])

        # Heading angle dim is: B x num_angle_per_cloud, where B is batch size
        # Flatten to (B * num_angle_per_cloud) x 1
        heading_angles = heading_angles.view(-1, 1)

        # B x heading_feature_dim
        heading_features = self.heading_backbone(heading_angles)

        # B x (pcd_feature_dim + heading_feature_dim)
        features = torch.hstack([pcd_features, heading_features])

        # B x n_cat_dist_bins
        output = self.head(features)

        output = self.sigmoid(output)

        return output
    

class FourierTerrainTraversabilityEncoder(nn.Module):

    """
    Model that predicts Fourier coefficients of the concentration parameters.
    Traversability distributions at required angles are obtained by querying
    these coefficients (dot product with basis functions).
    """

    def __init__(self, config):

        super().__init__()

        self.config = config

        # The order or frequency of the fourier basis functions
        # order = n ==> use 2n+1 basis functions e^{i k theta} for k in [-n, n]
        self.order = config['order']
        self.num_bins = config['num_bins']

        # Configs for point pillar encoder
        self.pillar_layer = PillarLayer(
            voxel_size = config['voxel_size'], 
            point_cloud_range = config['point_cloud_range'], 
            max_num_points = config['max_num_points'], 
            max_voxels = config['max_voxels']
        )
        self.pillar_encoder = PillarEncoder(
            voxel_size = config['voxel_size'], 
            point_cloud_range = config['point_cloud_range'], 
            in_channel=8, 
            out_channel=config['pointpillar_out_channel']
        )

        self.pcd_backbone = build_conv(config = config['pcd_backbone'])

        self.pooling = None
        if "pooling" in config:
            self.pooling = nn.MaxPool2d(
                kernel_size = config['pooling']['kernel_size'],
                stride = config['pooling']['stride'],
                padding = config['pooling']['padding']
            )

        self.head = build_head(config = config['head'])

    def fourier_basis(self, angles):

        f = [torch.ones_like(angles)]

        for i in range(1, self.order+1):

            f.append(torch.cos(i * angles))
            f.append(torch.sin(i * angles))

        f = torch.stack(f, dim=-1)

        return f

    def forward(self, input_dict):

        batched_pts = input_dict['batched_pts']

        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        
        # B x pcd_feature_dim x 1 x 1
        pcd_features = self.pcd_backbone(pillar_features)

        if self.pooling is not None:
            pcd_features = self.pooling(pcd_features)
        
        pcd_features = pcd_features.squeeze(-1).squeeze(-1)

        # Predict final Fourier Coefficients
        fourier_coefficients = self.head(pcd_features).view(pcd_features.shape[0], self.num_bins, -1)

        return fourier_coefficients
    

class TerrainTraversabilityEncoderAngleFree(nn.Module):

    """
    Model that predicts traversability distributions, without angle-conditioning.
    """

    def __init__(self, config):

        super().__init__()

        self.config = config

        # Configs for point pillar encoder
        self.pillar_layer = PillarLayer(
            voxel_size = config['voxel_size'], 
            point_cloud_range = config['point_cloud_range'], 
            max_num_points = config['max_num_points'], 
            max_voxels = config['max_voxels']
        )
        self.pillar_encoder = PillarEncoder(
            voxel_size = config['voxel_size'], 
            point_cloud_range = config['point_cloud_range'], 
            in_channel=8, 
            out_channel=config['pointpillar_out_channel']
        )

        self.num_bins = config['num_bins']

        # Backbone to encode the BEV feature map
        self.pcd_backbone = build_conv(config = config['pcd_backbone'])

        # Final MLP to predict the traversability distribution
        self.head = build_head(config = config['head'])

        # Sigmoid activation to produce valid concentration parameters
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict):

        batched_pts = input_dict['batched_pts']

        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        
        # B x pcd_feature_dim x 1 x 1
        pcd_features = self.pcd_backbone(pillar_features).squeeze(-1).squeeze(-1)

        # B x n_cat_dist_bins
        output = self.head(pcd_features)

        output = self.sigmoid(output)

        return output