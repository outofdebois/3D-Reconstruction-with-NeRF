""" Reconstruction of 3D models from LEGO dataset using a Neural Radiance Field (NeRF) approach """

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from trimesh.viewer.windowed import SceneViewer

from dataset import generate_rays
from model import NeuralRadianceField
from training import train_model
from testing import evaluate_model
from mesh import generate_mesh
from cloud import generate_point_cloud
from chamfer import compare_clouds

batch_size = 1024
device = 'cuda'
start_distance = 8.0
end_distance = 12.0
num_epochs = 10
learning_rate = 1e-3
lr_decay_factor = 0.5
sampling_bins = 100

dataset_path = 'C:/Users/user1/Desktop/Inżynierka/lego'

train_origins, train_directions, train_pixels = generate_rays(dataset_path, mode='train')
val_origins, val_directions, val_pixels = generate_rays(dataset_path, mode='val')

train_data_loader = DataLoader(torch.cat((
    torch.from_numpy(train_origins).reshape(-1, 3).type(torch.float),
    torch.from_numpy(train_directions).reshape(-1, 3).type(torch.float),
    torch.from_numpy(train_pixels).reshape(-1, 3).type(torch.float)
), dim=1), batch_size=batch_size, shuffle=True)

warmup_data_loader = DataLoader(torch.cat((
    torch.from_numpy(train_origins).reshape(100, 800, 800, 3)[:, 200:600, 200:600, :].reshape(-1, 3).type(torch.float),
    torch.from_numpy(train_directions).reshape(100, 800, 800, 3)[:, 200:600, 200:600, :].reshape(-1, 3).type(torch.float),
    torch.from_numpy(train_pixels).reshape(100, 800, 800, 3)[:, 200:600, 200:600, :].reshape(-1, 3).type(torch.float)
), dim=1), batch_size=batch_size, shuffle=True)

val_data_loader = DataLoader(torch.cat((
    torch.from_numpy(val_origins).reshape(-1, 3).type(torch.float),
    torch.from_numpy(val_directions).reshape(-1, 3).type(torch.float),
    torch.from_numpy(val_pixels).reshape(-1, 3).type(torch.float)
), dim=1), batch_size=batch_size, shuffle=False)

neural_model = NeuralRadianceField().to(device)
optimizer = torch.optim.Adam(neural_model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=lr_decay_factor)

warmup_loss = train_model(neural_model, optimizer, lr_scheduler, start_distance, end_distance, sampling_bins, 1, warmup_data_loader, device=device)
main_training_loss = train_model(neural_model, optimizer, lr_scheduler, start_distance, end_distance, sampling_bins, num_epochs, train_data_loader, device=device)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(warmup_loss, label="Warmup Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Warmup Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(main_training_loss, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.tight_layout()
plt.show()

torch.save(neural_model, 'model_nerf_lego')
trained_model = torch.load('model_nerf_lego').to(device)

test_image, mse, psnr = evaluate_model(
    trained_model,
    torch.from_numpy(val_origins[-1]).to(device).float(),
    torch.from_numpy(val_directions[-1]).to(device).float(),
    start_distance,
    end_distance,
    sampling_bins,
    chunk_size=10,
    ground_truth=val_pixels[-1].reshape(800, 800, 3)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(test_image)
axes[0].set_title("Rendered Image")
axes[0].axis('off')

axes[1].imshow(val_pixels[-1].reshape(800, 800, 3))
axes[1].set_title("Ground Truth")
axes[1].axis('off')

fig.text(0.5, 0.02, f"PSNR: {psnr:.2f}", ha='center', fontsize=12)
plt.tight_layout()
plt.show()

final_mesh = generate_mesh(trained_model, grid_resolution=100, bounding_box_size=1.5, device=device)
scene = final_mesh.scene()
SceneViewer(scene, start_loop=True, resolution=(1200, 1000))

point_cloud = generate_point_cloud(trained_model, grid_resolution=500, bounding_box_size=2., device=device, save_path="output_point_cloud.ply")
point_cloud.show()

file1 = "point_cloud.ply"
file2 = "output_point_cloud.ply"
compare_clouds(file1, file2)