import torch
import torch.utils.data as data
import numpy as np

import os
import glob

from my_utils.general import read_point_cloud


class DeformationDataset(data.Dataset):

    """
    Dataset of Tire Deformation distributions, when traversing various
    obstacles at different angles.

    Parameters:
    - base_dir (str): Base directory containing the data files.
    - angles_per_object (int): Number of angles per object.
    - max_points_per_object (int): Maximum number of points per object.
    - add_noise (bool): Whether to add noise to the data.
    - noise_config (dict): Configuration for the noise to be added.
    - num_bins (int): Number of bins for the deformation distribution.
    - min_deformation (float): Minimum deformation value.
    - max_deformation (float): Maximum deformation value.
    - mode (str): Mode of the dataset, [fourier, mlp, angle_free].

    Get Item Returns:
    - point_cloud (torch.Tensor): The point cloud data.
    - angles (torch.Tensor): The angles at which the object is traversed.
    - deformation_distribution (torch.Tensor): The deformation distributions 
        associated with the angles.
    """

    def __init__(self,
        base_dir,
        angles_per_object = 8,
        max_points_per_object = 1024,
        add_noise = True,
        noise_config = None,
        num_bins = 8,
        min_deformation = 0.2,
        max_deformation = 1.0,
        mode = "fourier"
    ):
        
        super().__init__()

        self.base_dir = base_dir
        # Each obstacle directory contains 1 object point cloud
        # and $(angles_per_object) collected distributions (with approach angle)
        self.obstacles_dirs = glob.glob(os.path.join(self.base_dir, "*"))

        self.angles_per_object = angles_per_object
        self.max_points_per_object = max_points_per_object
        self.add_noise = add_noise
        self.num_bins = num_bins
        self.min_deformation = min_deformation
        self.max_deformation = max_deformation

        if self.add_noise:
            assert noise_config is not None, "Noise config must be provided if add_noise is True"
        
        self.noise_config = noise_config

        assert mode in ["fourier", "mlp", "angle_free"], "Invalid mode, should be in [fourier, mlp, angle_free]"
        self.mode = mode


    def __len__(self):
        return len(self.obstacles_dirs)    

    def __getitem__(self, index):

        # Load the object point cloud
        pcd_path = os.path.join(self.obstacles_dirs[index], "obstacle.npy")

        pcd = read_point_cloud(pcd_path)

        # Extract points from within 0.5m
        valid_indices = torch.norm(pcd, dim=1) <= 0.5
        pcd = pcd[valid_indices].view(-1, 3)

        if self.add_noise:

            # randomly generate grass point cloud
            # Grass do not have collision volume in sim and thus are
            # not captured in the collected point clouds
            # We randomly generate grass noise to improve generalization
            # to real world
            
            # Any ground point can potentially be grass base position
            ground_indices = pcd[:, 2] <= 0.001
            ground_indices = ground_indices.nonzero().flatten()
            grass_indices = torch.rand(ground_indices.shape[0]) <= self.noise_config['p_grass']
            grass_indices = grass_indices.nonzero().flatten()
            grass_indices = ground_indices[grass_indices].flatten()
            grass_base_positions = pcd[grass_indices].view(-1, 3)
            base_xy = grass_base_positions[:, :2]

            # Grass height in range [min_grass_height, max_grass_height]
            grass_heights = torch.normal(
                mean = self.noise_config['mean_grass_height'], 
                std = self.noise_config['std_grass_height'],
                size = (grass_base_positions.shape[0],)
            )
            grass_heights = torch.clamp(grass_heights, min=self.noise_config['min_grass_height']).cuda()
            # grass_heights = self.noise_config['min_grass_height'] + torch.rand(grass_base_positions.shape[0]).cuda() \
            #     * (self.noise_config['max_grass_height'] - self.noise_config['min_grass_height'])
            
            # Generate grass points along each blade
            t = torch.linspace(0, 1, self.noise_config['points_per_blade']).unsqueeze(0).cuda()
            t = t.expand(grass_base_positions.shape[0], self.noise_config['points_per_blade'])  
            grass_heights = t * grass_heights.unsqueeze(1)

            # Add slight bending to each blade
            # For each blade, generate a random angle in [0, 2π)
            bend_angles = 2 * np.pi * torch.rand(grass_base_positions.shape[0])
            # Compute the 2D unit vector in the bending direction
            bend_directions = torch.stack((torch.cos(bend_angles), torch.sin(bend_angles)), dim=1).cuda()
            # Bending noise scale
            noise_scales = self.noise_config['max_noise_scale'] * torch.rand(grass_base_positions.shape[0]).cuda()
            # Bending offset increases with the square of t (to simulate gradual bending)
            # Expand noise_scales to match t's shape for elementwise multiplication
            bending_amplitude = noise_scales.unsqueeze(1) * (t ** 2)  # shape: (num_blades, points_per_blade)

            # Multiply with each blade's bending direction
            bending_offset = bending_amplitude.unsqueeze(2) * bend_directions.unsqueeze(1)  # shape: (num_blades, points_per_blade, 2)

            # Repeat base positions along the points dimension (shape: (num_blades, points_per_blade, 2))
            base_xy = base_xy.unsqueeze(1).expand(grass_base_positions.shape[0], self.noise_config['points_per_blade'], 2)

            # Compute the final noisy x, y coordinates for each point on each blade
            xy_coords = base_xy + bending_offset  # shape: (num_blades, points_per_blade, 2)

            # Combine x, y, z coordinates into a single tensor for the point cloud
            grass = torch.cat([xy_coords, grass_heights.unsqueeze(2)], dim=2)  # shape: (num_blades, points_per_blade, 3)
            grass = grass.view(-1, 3)  # flatten to (num_blades * points_per_blade, 3)

            pcd = torch.vstack([pcd, grass])

        # Randomly sample points if too many
        if pcd.shape[0] > self.max_points_per_object:
            sample_indices = torch.randperm(pcd.shape[0])[:self.max_points_per_object]
            pcd = pcd[sample_indices].view(-1, 3)

        # Load the angles and deformation distributions
        angles = []
        distributions = []

        for angle_index in range(self.angles_per_object):

            data_path = os.path.join(self.obstacles_dirs[index], f"angle_{angle_index}.npy")
            data = np.load(data_path)

            # Angle is the first element in the data array
            angle = data[0]
            angle = torch.tensor([angle]).cuda()

            # The rest are raw deformation values
            deformations = data[1:]
            deformations = torch.from_numpy(deformations.astype(np.float32)).cuda()

            # Keep only valid deformations [min_deformation, max_deformation]
            indices = deformations >= self.min_deformation
            deformations = deformations[indices].flatten()
            indices = deformations <= self.max_deformation
            deformations = deformations[indices].flatten()

            # Convert to histogram with $(num_bins) bins
            distribution = torch.histc(
                deformations,
                bins = self.num_bins,
                min = self.min_deformation,
                max = self.max_deformation
            )

            # Append to lists
            angles.append(angle)
            distributions.append(distribution)

        # For Fourier and MLP models, return all angles and their corresponding distributions
        if self.mode == "fourier" or self.mode == "mlp":
            
            # Convert histograms to valid categorical distributions
            for distribution in distributions:
                distribution /= distribution.sum()

            return pcd, angles, distributions
        
        else:       

            # For angle_free mode, collapse all angles' distributions into one.
            combined_distribution = torch.zeros_like(distributions[0]).cuda()
            for distribution in distributions:
                combined_distribution += distribution
            combined_distribution /= combined_distribution.sum()

            return pcd, combined_distribution