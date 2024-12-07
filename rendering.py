import torch

def calculate_transmittance(alphas):
    transmittance = torch.cumprod(alphas, dim=1)
    return torch.cat((torch.ones(transmittance.shape[0], 1, device=transmittance.device), transmittance[:, :-1]), dim=1)

def render_rays(model, origins, directions, t_min, t_max, bins=100, device='cpu', white_background=True):
    sample_points = torch.linspace(t_min, t_max, bins).to(device)
    intervals = torch.cat((sample_points[1:] - sample_points[:-1], torch.tensor([1e10], device=device)))
    sampled_positions = origins.unsqueeze(1) + sample_points.unsqueeze(0).unsqueeze(-1) * directions.unsqueeze(1)

    predicted_colors, densities = model.intersect(sampled_positions.reshape(-1, 3), directions.expand(sampled_positions.shape[1], sampled_positions.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    predicted_colors = predicted_colors.reshape((sampled_positions.shape[0], bins, 3))
    densities = densities.reshape((sampled_positions.shape[0], bins))

    alphas = 1 - torch.exp(- densities * intervals.unsqueeze(0))
    weights = calculate_transmittance(1 - alphas) * alphas

    if white_background:
        color_output = (weights.unsqueeze(-1) * predicted_colors).sum(1)
        accumulated_weights = weights.sum(-1)
        return color_output + 1 - accumulated_weights.unsqueeze(-1)
    else:
        color_output = (weights.unsqueeze(-1) * predicted_colors).sum(1)

    return color_output