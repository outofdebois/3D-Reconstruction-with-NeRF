import torch
import torch.nn as nn

class NeuralRadianceField(nn.Module):
    def __init__(self, position_encoding_levels=10, direction_encoding_levels=4, hidden_layer_size=256):
        super(NeuralRadianceField, self).__init__()

        self.position_block = nn.Sequential(
            nn.Linear(position_encoding_levels * 6 + 3, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )

        self.combined_block = nn.Sequential(
            nn.Linear(hidden_layer_size + position_encoding_levels * 6 + 3, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size + 1)
        )

        self.color_block = nn.Sequential(
            nn.Linear(hidden_layer_size + direction_encoding_levels * 6 + 3, hidden_layer_size // 2), nn.ReLU(),
            nn.Linear(hidden_layer_size // 2, 3), nn.Sigmoid()
        )

        self.position_encoding_levels = position_encoding_levels
        self.direction_encoding_levels = direction_encoding_levels

    def encode_positions(self, inputs, levels):
        encoded = [inputs]
        for level in range(levels):
            encoded.append(torch.sin(2 ** level * inputs))
            encoded.append(torch.cos(2 ** level * inputs))
        return torch.cat(encoded, dim=1)

    def forward(self, spatial_coords, ray_directions):
        position_encoded = self.encode_positions(spatial_coords, self.position_encoding_levels)
        direction_encoded = self.encode_positions(ray_directions, self.direction_encoding_levels)

        hidden_features = self.position_block(position_encoded)
        hidden_combined = self.combined_block(torch.cat((hidden_features, position_encoded), dim=1))

        sigma_values = hidden_combined[:, -1]
        rgb_values = self.color_block(torch.cat((hidden_combined[:, :-1], direction_encoded), dim=1))

        return rgb_values, torch.relu(sigma_values)