import torch
import numpy as np
import open3d as o3d
import struct


def read_point_cloud(filename):

    if ".ply" in filename:

        pcd = o3d.t.io.read_point_cloud(filename)
        points = pcd.point.positions.numpy()
    
    elif ".npy" in filename:

        points = np.load(filename)
        points = points.astype(np.float32)

    else:

        print(f"Unsupported file type")
        exit(1)

    points = torch.from_numpy(points).cuda()

    return points


def visPointCloud(pc_tensor: torch.Tensor):
    """
    Visualize an (N,3) torch.Tensor point cloud in Open3D.
    """
    # 1) Move to CPU and convert to NumPy
    pts = pc_tensor.detach().cpu().numpy()  # shape (N,3)

    # 2) Create an Open3D PointCloud and assign points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)  # vector of 3D points :contentReference[oaicite:0]{index=0}

    # 4) Launch the Open3D visualizer
    o3d.visualization.draw_geometries([pcd])  