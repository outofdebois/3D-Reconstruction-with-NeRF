import numpy as np
import trimesh
from scipy.spatial import cKDTree

def chamfer_distance(point_cloud1, point_cloud2):

    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)

    distances1, _ = tree1.query(point_cloud2, k=1)
    distances2, _ = tree2.query(point_cloud1, k=1)

    return np.mean(distances1) + np.mean(distances2)

def load_point_cloud(ply_file):

    mesh = trimesh.load(ply_file)
    if not isinstance(mesh, trimesh.points.PointCloud):
        raise ValueError(f"File {ply_file} don't have a point cloud.")
    return np.array(mesh.vertices)

def compare_clouds(file1, file2):

    points1 = load_point_cloud(file1)
    points2 = load_point_cloud(file2)

    distance = chamfer_distance(points1, points2)
    print(f"Chamfer Distance between {file1} and {file2}: {distance}")
    return distance