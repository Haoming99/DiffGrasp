import open3d as o3d

# Load the two PCD files
pcd1 = o3d.io.read_point_cloud("tmp_data/complete_pc.pcd")
pcd2 = o3d.io.read_point_cloud("tmp_data/complete_pc_vae.pcd")

# Translate the second point cloud to the right for better visualization
#pcd2.translate((0.5, 0, 0))  # Adjust the translation as needed

pcd1.paint_uniform_color([0, 0, 1])
pcd2.paint_uniform_color([1, 0, 0])
# Create a visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add both point clouds to the visualizer
vis.add_geometry(pcd1)
vis.add_geometry(pcd2)

# Run the visualizer
vis.run()
vis.destroy_window()
