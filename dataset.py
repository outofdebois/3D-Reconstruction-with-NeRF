import json
import numpy as np
import os
import imageio.v2 as imageio

def generate_rays(data_directory, mode='train'):
    json_file_path = os.path.join(data_directory, f'transforms_{mode}.json')
    with open(json_file_path, 'r') as file:
        transform_data = json.load(file)

    frames = transform_data['frames']
    total_images = len(frames)
    camera_fov = transform_data['camera_angle_x']
    focal_length_pixels = 0.5 / np.tan(camera_fov / 2)

    pose_matrices = np.zeros((total_images, 4, 4))
    image_data = []

    for index, frame in enumerate(frames):
        pose_matrices[index] = np.array(frame['transform_matrix'], dtype=np.float32)
        image_path = os.path.join(data_directory, frame['file_path'][2:] + '.png')
        image = imageio.imread(image_path) / 255.0
        image_data.append(image[None, ...])

    images_combined = np.concatenate(image_data)
    image_height, image_width = images_combined.shape[1:3]

    if images_combined.shape[3] == 4:
        images_combined = images_combined[..., :3] * images_combined[..., -1:] + (1 - images_combined[..., -1:])

    ray_origins = np.zeros((total_images, image_height * image_width, 3))
    ray_directions = np.zeros((total_images, image_height * image_width, 3))
    pixel_targets = images_combined.reshape((total_images, image_height * image_width, 3))

    for index in range(total_images):
        camera_matrix = pose_matrices[index]

        pixel_u, pixel_v = np.meshgrid(np.arange(image_width), np.arange(image_height))
        direction_vectors = np.stack([
            pixel_u - image_width / 2,
            -(pixel_v - image_height / 2),
            -np.ones_like(pixel_u) * focal_length_pixels
        ], axis=-1)

        direction_vectors = (camera_matrix[:3, :3] @ direction_vectors[..., None]).squeeze(-1)
        direction_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=-1, keepdims=True)

        ray_directions[index] = direction_vectors.reshape(-1, 3)
        ray_origins[index] += camera_matrix[:3, 3]

    return ray_origins, ray_directions, pixel_targets