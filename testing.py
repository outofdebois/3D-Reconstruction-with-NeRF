import torch
import numpy as np
from rendering import render_rays

def calculate_psnr(mse_error):
    return 20 * np.log10(1 / np.sqrt(mse_error))

@torch.no_grad()
def evaluate_model(model, ray_origins, ray_directions, t_min, t_max, bins=100, chunk_size=10, height=800, width=800,
                   ground_truth=None):
    ray_origin_chunks = ray_origins.chunk(chunk_size)
    ray_direction_chunks = ray_directions.chunk(chunk_size)

    reconstructed_image = []
    for origin_batch, direction_batch in zip(ray_origin_chunks, ray_direction_chunks):
        batch_render = render_rays(model, origin_batch, direction_batch, t_min, t_max, bins=bins,
                                   device=origin_batch.device)
        reconstructed_image.append(batch_render)

    reconstructed_image = torch.cat(reconstructed_image)
    final_image = reconstructed_image.reshape(height, width, 3).cpu().numpy()

    if ground_truth is not None:
        mse_error = ((final_image - ground_truth) ** 2).mean()
        psnr_score = calculate_psnr(mse_error)
        return final_image, mse_error, psnr_score
    else:
        return final_image