import torch
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

def filter_out_noise(points, colors, radius=0.01, min_neighbors=200):

    nbrs = NearestNeighbors(radius=radius).fit(points)
    neighbor_counts = nbrs.radius_neighbors(points, return_distance=False)

    valid_indices = [i for i, neighbors in enumerate(neighbor_counts) if len(neighbors) >= min_neighbors]
    filtered_points = points[valid_indices]
    filtered_colors = colors[valid_indices]

    return filtered_points, filtered_colors

def generate_point_cloud(neural_model, grid_resolution=100, bounding_box_size=1.5, chunk_size=500000, device='cpu', save_path=None):

    grid_x = torch.linspace(-bounding_box_size, bounding_box_size, grid_resolution)
    grid_y = torch.linspace(-bounding_box_size, bounding_box_size, grid_resolution)
    grid_z = torch.linspace(-bounding_box_size, bounding_box_size, grid_resolution)

    x_grid, y_grid, z_grid = torch.meshgrid((grid_x, grid_y, grid_z), indexing='ij')
    grid_points = torch.cat((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), z_grid.reshape(-1, 1)), dim=1).to(device)

    valid_points = []
    valid_colors = []

    for i in range(0, grid_points.size(0), chunk_size):
        chunk = grid_points[i:i + chunk_size]
        with torch.no_grad():
            predicted_colors, density_values = neural_model.forward(
                chunk,
                torch.zeros_like(chunk)
            )

        density_values = density_values.squeeze()
        valid_indices = density_values > 8 * density_values.mean()

        valid_points.append(chunk[valid_indices].cpu().numpy())
        valid_colors.append(predicted_colors[valid_indices].cpu().numpy())

    valid_points = np.concatenate(valid_points, axis=0)
    valid_colors = np.concatenate(valid_colors, axis=0)
    valid_colors = (valid_colors * 255).astype(np.uint8)

    valid_points, valid_colors = filter_out_noise(valid_points, valid_colors, radius=0.05, min_neighbors=10)
    point_cloud = trimesh.points.PointCloud(vertices=valid_points, colors=valid_colors)

    if save_path:
        point_cloud.export(save_path)
        print(f"Chmura punktów zapisana do pliku: {save_path}")

    return point_cloud