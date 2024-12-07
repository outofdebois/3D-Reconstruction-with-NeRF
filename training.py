from tqdm import tqdm
from rendering import render_rays
import torch

def train_model(model, optimizer, scheduler, t_min, t_max, bins, epochs, data_loader, device='cpu'):
    loss_history = []

    for epoch in range(epochs):
        for batch_data in tqdm(data_loader):
            ray_origins = batch_data[:, :3].to(device)
            ray_directions = batch_data[:, 3:6].to(device)
            target_pixels = batch_data[:, 6:].to(device)

            rendered_pixels = render_rays(model, ray_origins, ray_directions, t_min, t_max, bins=bins, device=device)
            loss = ((rendered_pixels - target_pixels) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        scheduler.step()
        torch.save(model.cpu(), 'nerf_trained_model')
        model.to(device)

    return loss_history