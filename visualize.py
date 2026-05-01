import open3d as o3d
import numpy as np
import os

OUTPUT_DIR = 'outputs'

def draw_scene(pointcloud, filename="scene"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    colors = np.zeros((len(pointcloud), 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f"./{OUTPUT_DIR}/{filename}.ply", pcd)

    print(f"[INFO] Saved: {filename}.ply")
