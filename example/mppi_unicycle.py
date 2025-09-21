import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_Fourier_basis(angles, order=3):

    f = [torch.ones_like(angles)]

    for i in range(1, order+1):

        f.append(torch.cos(i * angles))
        f.append(torch.sin(i * angles))

    f = torch.stack(f, dim=-1)

    return f


def compute_VaR(p, bin_centers, alpha):

    cdf = torch.cumsum(p, dim=1)
    var_idx = (cdf >= alpha).float().argmax(dim=1).long()  # shape: (B,)
    VaR = bin_centers[var_idx]

    return VaR.flatten(), var_idx.flatten()


def compute_CVaR(p, bin_centers, alpha):

    VaR, var_idx = compute_VaR(p, bin_centers, alpha)

    # To compute CVaR, we take the weighted average of outcomes in the tail for each batch.
    # Create a tensor of bin indices for comparison.
    bin_indices = torch.arange(p.shape[1]).unsqueeze(0).expand(p.shape[0], p.shape[1]).cuda()  # shape: (B, N)

    # Create a boolean mask indicating which bins are in the tail (i.e., index >= VaR index for that batch)
    mask = (bin_indices >= var_idx.unsqueeze(1))  # shape: (B, N)

    # Compute the total tail probability mass for each batch.
    tail_mass = (p * mask.float()).sum(dim=1)  # shape: (B,)

    # Compute the weighted sum of outcomes for the tail.
    weighted_sum = (p * bin_centers.unsqueeze(0) * mask.float()).sum(dim=1)  # shape: (B,)

    # CVaR is the weighted average over the tail.
    CVaR = weighted_sum / tail_mass  # shape: (B,)

    return CVaR


def unicycle_dynamics(x, u, dt):
    """
    Unicycle dynamics model.
    
    Args:
        x (torch.Tensor): Current state tensor with shape [batch, 3] representing (x, y, theta).
        u (torch.Tensor): Control input tensor with shape [batch, 2] representing (v, omega).
        dt (float): Time step.
    
    Returns:
        torch.Tensor: Next state tensor with shape [batch, 3].
    """
    # Allocate tensor for next state
    x_next = torch.empty_like(x).cuda()
    # x and y update using forward velocity and current heading
    x_next[:, 0] = x[:, 0] + u[:, 0] * torch.cos(x[:, 2]) * dt
    x_next[:, 1] = x[:, 1] + u[:, 0] * torch.sin(x[:, 2]) * dt
    # theta update using angular velocity
    x_next[:, 2] = x[:, 2] + u[:, 1] * dt
    return x_next


def rollout_trajectory(x_init, U, dt):
    """
    Rollout candidate control sequences to produce full trajectories.
    
    Args:
        x_init (torch.Tensor): Initial state tensor of shape [3].
        U (torch.Tensor): Candidate control sequences of shape [K, H, 2] where 
                        K is the number of candidates and H is the time horizon.
        dt (float): Time step.
    
    Returns:
        traj (torch.Tensor): Trajectories tensor of shape [K, H+1, 3].
    """
    K, H, _ = U.shape
    # Preallocate tensor for trajectories: each row corresponds to a candidate's trajectory.
    traj = torch.empty(K, H+1, 3, device=U.device)
    
    # Set the initial state for all trajectories
    traj[:, 0, :] = x_init.unsqueeze(0).repeat(K, 1)
    
    # Rollout over the entire horizon
    for h in range(H):
        traj[:, h+1, :] = unicycle_dynamics(traj[:, h, :], U[:, h, :], dt)
        
    return traj


def wheel_rollouts(traj: torch.Tensor, d: float):
    """
    Given a centre‐of‐robot rollout 'traj' [..., (x,y,θ)],
    compute the left‐ and right‐wheel rollouts [..., (x,y)].
    """
    # unpack
    x = traj[..., 0]
    y = traj[..., 1]
    theta = traj[..., 2]
    
    # unit normal to heading (points to robot's left)
    # n = [ -sinθ,  cosθ ]
    nx = -torch.sin(theta)
    ny =  torch.cos(theta)
    
    # left wheel = centre + n*d_left
    left_x = x - nx * d
    left_y = y - ny * d
    left  = torch.stack([left_x, left_y], dim=-1)
    
    # right wheel = centre - n*d_right
    right_x = x + nx * d
    right_y = y + ny * d
    right   = torch.stack([right_x, right_y], dim=-1)
    
    return left, right


class MPPI_Unicycle:

    def __init__(self, mode='ours', device='cuda'):

        self.mode = mode
        self.device = device

        # MPPI parameters
        self.dt = 0.1
        self.H = 60
        self.num_samples = 10000
        self.lambda_ = 100.0
        self.sigma = 1.0
        self.num_iterations = 1

        # Control limits
        self.control_lb = torch.tensor([-2, -torch.pi]).float().to(self.device)
        self.control_ub = torch.tensor([ 2,  torch.pi]).float().to(self.device)

        # Map configs
        self.workspace_size = 0.1
        self.alpha = 0.9 # For CVaR computation
        self.map_resolution = 0.05
        self.map_boundary = torch.tensor([-20, -20, 20, 20]).float().to(self.device)
        self.x_bins = int((self.map_boundary[2] - self.map_boundary[0]) // self.map_resolution)
        self.y_bins = int((self.map_boundary[3] - self.map_boundary[1]) // self.map_resolution)

        # Load maps
        self.obstacle_map = np.load("example/obstacle_map.npy")
        self.elevation_map = np.load("example/elevation_map.npy")
        self.coeff_map = np.load("example/coeff_map.npy")
        self.obstacle_map = torch.from_numpy(self.obstacle_map).float().to(self.device)
        self.elevation_map = torch.from_numpy(self.elevation_map).float().to(self.device)
        self.coeff_map = torch.from_numpy(self.coeff_map).float().to(self.device)

        # the planning goal of the robot
        self.target = torch.tensor([0.0, 5.5]).float().to(self.device)

        # Vehicle parameters
        self.wheel_dist = 0.25 # distance between left and right wheels

        # Nominal control sequence (initially forward motion)
        self.u_nominal = torch.tensor([[1.0, 0.0] for _ in range(self.H)]).float().to(self.device)

    def evaluate_rollout(self, traj):
        
        """
        Evaluate the rollout trajectory using a few costmaps

        Args:
            traj: trajectory rollout using unicycle model

        Returns:
            Score for every trajectory in the batch
        """ 

        # Cumulative squared Euclidean distance from the target
        # cost_pos = torch.sum((traj[:, :, :2] - self.target)**2, dim=(1, 2))
        cost_pos = torch.norm(self.target - traj[:, :, :2], dim=-1)
        cost_pos = torch.sum(cost_pos, dim=-1)

        # compute corresponding indices for base pos
        indices_x = ((traj[:, :, 0] - self.map_boundary[0]) / self.map_resolution).long().flatten()
        indices_y = ((traj[:, :, 1] - self.map_boundary[1]) / self.map_resolution).long().flatten()

        # Compute left and right wheel trajectories
        left, right = wheel_rollouts(traj, self.wheel_dist)
        left_indices_x = ((left[:, :, 0] - self.map_boundary[0]) / self.map_resolution).long().flatten()
        left_indices_y = ((left[:, :, 1] - self.map_boundary[1]) / self.map_resolution).long().flatten()
        right_indices_x = ((right[:, :, 0] - self.map_boundary[0]) / self.map_resolution).long().flatten()
        right_indices_y = ((right[:, :, 1] - self.map_boundary[1]) / self.map_resolution).long().flatten()

        # cost for traversing lethal obstacle
        base_obs_cost = self.obstacle_map[indices_x, indices_y].view(traj.shape[0], traj.shape[1])
        left_obs_cost = self.obstacle_map[left_indices_x, left_indices_y].view(traj.shape[0], traj.shape[1])
        right_obs_cost = self.obstacle_map[right_indices_x, right_indices_y].view(traj.shape[0], traj.shape[1])
        obs_cost = torch.stack([base_obs_cost, left_obs_cost, right_obs_cost], dim=-1).max(dim=-1).values
        obs_cost = obs_cost.sum(dim = -1)

        left_elev_cost = self.elevation_map[left_indices_x, left_indices_y].view(traj.shape[0], traj.shape[1])
        right_elev_cost = self.elevation_map[right_indices_x, right_indices_y].view(traj.shape[0], traj.shape[1])

        # Elevation cost
        if self.mode == 'elevation':

            elev_cost = torch.stack([left_elev_cost, right_elev_cost], dim=-1).max(dim=-1).values
            elev_cost = elev_cost.max(dim = -1).values

            elev_cost = (elev_cost - elev_cost.min()) / \
                (elev_cost.max() - elev_cost.min() + 1e-4)

            return cost_pos + 20 * obs_cost + 200 * elev_cost
        
        elif self.mode == 'ours':
            
            '''
            Now this does not work so Problem is here
            '''
            approach_angles = traj[:, :, 2].flatten().unsqueeze(-1) + torch.pi

            coeff_left = self.coeff_map[left_indices_x, left_indices_y]
            coeff_right = self.coeff_map[right_indices_x, right_indices_y]

            basis_functions = compute_Fourier_basis(approach_angles)
            basis_functions = torch.cat([basis_functions, basis_functions], dim=0)

            # compute distributions
            coeffs = torch.cat([coeff_left, coeff_right], dim=0)
            distributions = torch.einsum("bik,bjk->bij", basis_functions, coeffs).squeeze(1)
            distributions = torch.nn.functional.sigmoid(distributions)
            distributions_left = distributions[:distributions.shape[0]//2]
            distributions_right = distributions[distributions.shape[0]//2:]

            # Compute the CVaRs
            distributions_left = distributions_left / distributions_left.sum(dim=-1, keepdim=True)
            distributions_right = distributions_right/ distributions_right.sum(dim=-1, keepdim=True)
            bin_edges = torch.linspace(0.2, 1.0, 9).cuda()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            CVaRs_left = compute_CVaR(distributions_left, bin_centers, self.alpha).view(traj.shape[0], traj.shape[1])
            CVaRs_right = compute_CVaR(distributions_right, bin_centers, self.alpha).view(traj.shape[0], traj.shape[1])

            CVaRs_left = torch.where(
                left_elev_cost <= 0.01, torch.zeros_like(CVaRs_left), CVaRs_left
            )
            CVaRs_right = torch.where(
                right_elev_cost <= 0.01, torch.zeros_like(CVaRs_right), CVaRs_right
            )

            CVaRs_left = torch.where(
                left_elev_cost >= 0.1, torch.ones_like(CVaRs_left), CVaRs_left
            )
            CVaRs_right = torch.where(
                right_elev_cost >= 0.1, torch.ones_like(CVaRs_right), CVaRs_right
            )

            CVaR_cost = torch.stack([
                CVaRs_left, # + 0.1 * left_elev_cost, 
                CVaRs_right # + 0.1 * right_elev_cost
            ], dim=-1).max(dim=-1).values
            CVaR_cost = CVaR_cost.max(dim=-1).values

            CVaR_cost = (CVaR_cost - CVaR_cost.min()) / \
                (CVaR_cost.max() - CVaR_cost.min() + 1e-4)

            return cost_pos + 20 * obs_cost + 420 * CVaR_cost

        return cost_pos + 20 * obs_cost

    def mppi_update(self, x_init, U_nominal):
        """
        Perform one MPPI update iteration on the nominal control sequence.
        
        Args:
            x_init (torch.Tensor): Initial state of shape [3].
            U_nominal (torch.Tensor): Nominal control sequence of shape [H, 2].
            dt (float): Time step.
            num_samples (int): Number of candidate control sequence samples.
            lambda_ (float): Temperature parameter.
            sigma (float): Standard deviation for additive control noise.
        
        Returns:
            U_updated (torch.Tensor): Updated nominal control sequence of shape [H, 2].
            costs (torch.Tensor): Total cost for each candidate trajectory, shape [num_samples].
        """
        H, control_dim = U_nominal.shape
        U_nominal_expanded = U_nominal.unsqueeze(0).expand(self.num_samples, H, control_dim)
        noise = self.sigma * torch.randn(self.num_samples, H, control_dim).cuda()
        U_candidates = U_nominal_expanded + noise
        U_candidates = torch.clamp(U_candidates, min=self.control_lb, max=self.control_ub)
        traj = rollout_trajectory(x_init, U_candidates, self.dt)

        costs = self.evaluate_rollout(traj)
        
        weights = torch.exp(-costs / self.lambda_)
        weights = weights / (weights.sum() + 1e-12)
        
        weighted_noise = (weights.unsqueeze(1).unsqueeze(2) * noise).sum(dim=0)
        U_updated = U_nominal + weighted_noise
        U_updated = torch.clamp(U_updated, min=self.control_lb, max=self.control_ub)
        return U_candidates, U_updated, traj, costs

    def run_experiment(self):

        # Plotting costmaps as the background
        if self.mode == 'elevation':

            plt.imshow(self.elevation_map.T.cpu().numpy())

        elif self.mode == 'ours':   

            coefficients = self.coeff_map.view(-1, self.coeff_map.shape[-2], self.coeff_map.shape[-1]).to(self.device)

            # Expected traversing angle through the narrow passage ways (over the obstacles)
            t = -0.5*torch.pi
            angles = t * torch.ones((coefficients.shape[0], 1)).to(self.device)
            basis = compute_Fourier_basis(angles, order=3)
            distributions = torch.einsum("bik,bjk->bij", basis, coefficients).squeeze(1)
            distributions = torch.nn.functional.sigmoid(distributions)
            distributions = distributions / distributions.sum()
            bin_edges = torch.linspace(0.2, 1.0, 9).cuda()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            CVaRs = compute_CVaR(distributions, bin_centers, self.alpha)
            CVaRs = CVaRs.view(self.coeff_map.shape[0], self.coeff_map.shape[1])
            CVaRs = torch.where(
                self.elevation_map <= 0.01,
                torch.zeros_like(CVaRs),
                CVaRs
            )

            CVaRs = torch.where(
                self.elevation_map > 0.1,
                CVaRs.max()*torch.ones_like(CVaRs),
                CVaRs
            )

            plt.imshow(CVaRs.T.cpu().numpy())

        # Plot target
        plt.scatter(
            [(self.target[0].item() - self.map_boundary[0].item()) / self.map_resolution],
            [(self.target[1].item() - self.map_boundary[1].item()) / self.map_resolution],
            s = 50, c = 'red'
        )

        # Tracker for processing time (assuming pre-computed coefficients)
        runtimes = []

        # repeat the experiment for 10 trials
        for trial in range(10):

            # Spawn the robot
            x_world = torch.tensor([0.0, 0.5, 0.5*torch.pi]).float().to(self.device) # World frame
            x_robot = torch.tensor([0.0, 0.0, 0.0]).float().to(self.device) # Robot frame

            # MPPI iterations
            u_current = self.u_nominal.clone()
            traj = None
            costs = None
            steps = 0

            trajectory_x = []
            trajectory_y = []

            while steps < 100:
                
                # Convert current robot frame pose to world frame
                x_world[0] = -x_robot[1]
                x_world[1] = x_robot[0] + 0.5
                x_world[2] = x_robot[2] + 0.5*torch.pi

                # Record current position in trajectory
                trajectory_x.append(x_world[0].cpu().item())
                trajectory_y.append(x_world[1].cpu().item())

                # Check proximity to goal
                if torch.norm(self.target - x_world[:2]) <= 0.5:
                    break

                start = time.time()

                for i in range(self.num_iterations):

                    u_candidate, u_current, traj, costs = self.mppi_update(
                        x_world, u_current
                    )

                runtimes.append(time.time() - start)

                # Apply the first action and shift the control sequence
                action = u_current[0]
                x_robot[0] = x_robot[0] + action[0] * torch.cos(x_robot[2]) * self.dt
                x_robot[1] = x_robot[1] + action[0] * torch.sin(x_robot[2]) * self.dt
                x_robot[2] = x_robot[2] + action[1] * self.dt

                # Shift the control sequence
                u_current = torch.vstack([u_current[1:], u_current[-1:]])
                
                steps += 1

            trajectory_x = np.array(trajectory_x)
            trajectory_y = np.array(trajectory_y)

            # Plot trajectory
            plt.plot(
                (trajectory_x - self.map_boundary[0].item()) / self.map_resolution,
                (trajectory_y - self.map_boundary[1].item()) / self.map_resolution,
                c = 'white', linewidth = 2
            )
        
        plt.xlim(320, 480)
        plt.ylim(395, 525)
        plt.show()

        avg_runtime = sum(runtimes) / len(runtimes)
        print(f"Runtime: {avg_runtime} s")
        print(f"Frequency: {1 / avg_runtime} Hz")


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    mppi = MPPI_Unicycle(mode='elevation', device='cuda')
    mppi.run_experiment()