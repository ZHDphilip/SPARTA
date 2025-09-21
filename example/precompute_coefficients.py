import torch
import numpy as np
import yaml
import open3d as o3d
import matplotlib.pyplot as plt

from my_utils.training import load_model


class MapWorker:

    def __init__(self, model_config_path, device="cuda"):

        self.device = device
        self.load_model_from_config(model_config_path, device=device)

        # Define the map parameters
        self.workspace_size = 0.1
        self.alpha = 0.9 # For CVaR computation
        self.map_resolution = 0.05
        self.map_boundary = torch.tensor([-20, -20, 20, 20]).float().to(self.device)
        self.x_bins = int((self.map_boundary[2] - self.map_boundary[0]) // self.map_resolution)
        self.y_bins = int((self.map_boundary[3] - self.map_boundary[1]) // self.map_resolution)

    def load_model_from_config(self, config_path, device="cuda"):

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model, _, _ = load_model(
            config = self.config,
            training_set_length = 10000,
            load_model_state_dict = True,
            load_optimizer_state_dict = False,
            load_scheduler_state_dict = False,
            device = device
        )

        self.model.eval()
        self.model.to(device)

    def compute_save_maps(self):

        obstacle_map = torch.zeros((self.x_bins, self.y_bins)).to(self.device)
        x_centres = torch.arange(self.x_bins, dtype=torch.float32).to(self.device) * self.map_resolution + self.map_boundary[0] + self.map_resolution/2
        y_centres = torch.arange(self.y_bins, dtype=torch.float32).to(self.device) * self.map_resolution + self.map_boundary[1] + self.map_resolution/2
        xx, yy = torch.meshgrid(x_centres, y_centres, indexing='ij')  # 'ij' = matrix indexing
        centers = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)

        '''
        Initialize simple obstacle map
                             5 M
        +--------------------------------------------+ 6 M
        |                                            |
        |                 GOAL (0, 5)                |
        |      -1.2     -0.2      0.2      1.2       |
        |++++++++|        |++++++++|        |++++++++| 4
        |++++++++|        |++++++++|        |++++++++|
        |++++++++|        |++++++++|        |++++++++|
        |++++++++|  Safe  |++++++++|  Risk  |++++++++|
        |++++++++|        |++++++++|        |++++++++|
        |++++++++|        |++++++++|        |++++++++|
        |++++++++|        |++++++++|        |++++++++| 2
        |                                            |
        |                                            |
        |               Start (0, 0.5)               |
        +--------------------------------------------+ 0
        '''
        is_obs = (
            (
                ((centers[..., 0] >= -2.5) & (centers[..., 0] <= -1.2)) | \
                ((centers[..., 0] >= -0.2) & (centers[..., 0] <= 0.2)) | \
                ((centers[..., 0] >= 1.2) & (centers[..., 0] <= 2.5))
            ) & (
                (centers[..., 1] >= 2) & (centers[..., 1] <= 4)
            )
        ) | (
            centers[..., 0] < -2.5
        ) | (
            centers[..., 0] > 2.5
        ) | (
            centers[..., 1] < 0
        ) | (
            centers[..., 1] > 6
        )

        obstacle_map = torch.where(
            is_obs, 1, 0
        ).view(self.x_bins, self.y_bins)

        # Load the safe/risky point clouds, which are generated similar to the physical obstacles
        # used in the real world experiments
        safe_pcd = np.load("./example/safe_pcd.npy")
        risky_pcd = np.load("./example/risky_pcd.npy")
        safe_pcd = torch.from_numpy(safe_pcd).float().to(self.device)
        risky_pcd = torch.from_numpy(risky_pcd).float().to(self.device)

        # Place them at their respective locations in the map
        safe_pcd[:, 0] -= 1.2
        safe_pcd[:, 1] += 2.0
        risky_pcd[:, 0] += 0.2
        risky_pcd[:, 1] += 2.0

        # Construct full point cloud map
        map_pcd = torch.vstack([safe_pcd, risky_pcd])

        # Construct elevation map
        
        # Map (x, y) coordinates to grid indices
        x_indices = ((map_pcd[:, 0] - self.map_boundary[0]) / self.map_resolution).long()
        y_indices = ((map_pcd[:, 1] - self.map_boundary[1]) / self.map_resolution).long()

        # Compute linear indices for unique grid cell id
        linear_indices = x_indices * self.y_bins + y_indices

        # Construct elevation map
        elevation_map = torch.full((self.x_bins * self.y_bins,), 0.0, device='cuda')
        elevation_map = torch.scatter_reduce(elevation_map, 0, linear_indices, map_pcd[:, 2], reduce='amax')
        elevation_map = elevation_map.view(self.x_bins, self.y_bins)

        # Compute Fourier Coefficient Map

        # obtain all terrain patches for model query
        query_pcds = []
        obstacle_mask = torch.full(elevation_map.shape, False, dtype=torch.bool).to(self.device)

        # Iterate over each row in the grid
        for i in range(self.x_bins):

            # Find all grids in the current row containing obstacles,
            # i.e. with elevation >= tolerance (0.01)
            with_obstacle = elevation_map[i] >= 0.01
            with_obstacle = with_obstacle.nonzero().flatten()

            # get grid centers
            grid_centers = torch.hstack([
                self.map_boundary[0] + (i+0.5) * self.map_resolution * torch.ones_like(with_obstacle).view(-1, 1),
                self.map_boundary[1] + (with_obstacle.view(-1, 1) + 0.5) * self.map_resolution,
                torch.zeros_like(with_obstacle).view(-1, 1)
            ])

            if grid_centers.shape[0] == 0:
                continue

            sub_map_indices = (map_pcd[:, 0] >= self.map_boundary[0] + (i-1.1) * self.map_resolution) & \
                (map_pcd[:, 0] <= self.map_boundary[0] + (i+2.1) * self.map_resolution)
            sub_map = map_pcd[sub_map_indices].view(-1, 3)

            for point in grid_centers:

                indices = torch.norm(sub_map - point, dim=-1) <= self.workspace_size
                pcd = sub_map[indices].view(-1, 3)
                indices = torch.randperm(pcd.shape[0])[:self.config['dataset']['max_points_per_object']]
                pcd = pcd[indices].view(-1, 3) - point
                min_z = pcd[:, 2].min()
                pcd[:, 2] = pcd[:, 2] - min_z
                query_pcds.append(pcd*5)

            obstacle_mask[i][with_obstacle] = True

        query_pcds = torch.stack(query_pcds, dim=0).to(self.device)

        # Mark lethal obstacles with high elevation (0.5)
        elevation_map[obstacle_map.bool()] = 0.5

        # Query the model to obtain Fourier Coefficients
        with torch.no_grad():
            
            input_dict = {'batched_pts': query_pcds}
            fourier_coefficients = self.model(input_dict)

        # fill fourier coefficient map
        coeff_map = torch.zeros((
            self.x_bins, self.y_bins, self.model.num_bins, self.model.order * 2 + 1
        )).to(self.device)

        coeff_map[obstacle_mask] = fourier_coefficients

        # Save the maps
        np.save(
            "./example/obstacle_map.npy", obstacle_map.cpu().numpy()
        )
        np.save(
            "./example/elevation_map.npy", elevation_map.cpu().numpy()
        )
        np.save(
            "./example/coeff_map.npy", coeff_map.cpu().numpy()
        )


if __name__ == "__main__":

    worker = MapWorker(
        model_config_path = "./config/tire_deformation_fourier_new.yaml",
        device = "cuda"
    )

    worker.compute_save_maps()