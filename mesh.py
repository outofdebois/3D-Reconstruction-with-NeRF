import torch
import mcubes
import trimesh
import numpy as np

def generate_mesh(neural_model, grid_resolution=100, bounding_box_size=1.5, device='cpu'):
    grid_x = torch.linspace(-bounding_box_size, bounding_box_size, grid_resolution)
    grid_y = torch.linspace(-bounding_box_size, bounding_box_size, grid_resolution)
    grid_z = torch.linspace(-bounding_box_size, bounding_box_size, grid_resolution)

    x_grid, y_grid, z_grid = torch.meshgrid((grid_x, grid_y, grid_z))
    grid_points = torch.cat((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), z_grid.reshape(-1, 1)), dim=1)

    with torch.no_grad():
        _, density_values = neural_model.forward(grid_points.to(device), torch.zeros_like(grid_points).to(device))

    density_reshaped = density_values.cpu().numpy().reshape(grid_resolution, grid_resolution, grid_resolution)
    vertices, faces = mcubes.marching_cubes(density_reshaped, 25 * np.mean(density_reshaped))
    final_mesh = trimesh.Trimesh(vertices / grid_resolution, faces)

    return final_mesh