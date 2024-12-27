from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import subprocess
import rospy
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs  # **Do not use geometry_msgs. Use this instead for PoseStamped
import signal
import matplotlib.pyplot as plt

import torch
from Completion.Wen.models.conv_decoder import LocalDecoderAttn
from Completion.Wen.models.local_encoder import LocalPoolPointnet
from Completion.Wen.models.diff_cli import UNet2DModelCLI, AutoencoderKLCLI
from diffusers import PNDMScheduler
from Completion.Wen.models.diffusion3x_c_torch import Diffusion3XC



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point



def handler(signum, frame):
    res = input("Ctrl-c was pressed. Do you really want to exit? [Y]/n ")
    if res == 'y':
        exit(1)


def transform_pose(input_pose, from_frame, to_frame):
    # **Assuming /tf2 topic is being broadcasted
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = input_pose
    pose_stamped.header.frame_id = from_frame
    # pose_stamped.header.stamp = rospy.Time.now()

    rospy.sleep(2)
    try:
        # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(1))
        return output_pose_stamped.pose

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise


def publish_tf(px, py, pz, rx, ry, rz, rw, par_frame="base_link", child_frame="goal_frame"):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = par_frame
    static_transformStamped.child_frame_id = child_frame

    static_transformStamped.transform.translation.x = float(px)
    static_transformStamped.transform.translation.y = float(py)
    static_transformStamped.transform.translation.z = float(pz)

    static_transformStamped.transform.rotation.x = float(rx)
    static_transformStamped.transform.rotation.y = float(ry)
    static_transformStamped.transform.rotation.z = float(rz)
    static_transformStamped.transform.rotation.w = float(rw)

    broadcaster.sendTransform(static_transformStamped)
    rospy.sleep(5)

    return True


def str_to_np(grasp_string):
    grasp_pose = []
    for line in g.splitlines():
        d = line.split(':')
        nums = d[1].split()
        if d[0] == "grasp width" or d[0] == "grasp surface": continue
        # print ()
        # print ([float(n) for n in nums])
        grasp_pose.append([float(n) for n in nums])

    grasp_pose = np.asarray(grasp_pose)
    return grasp_pose


def read_grasps():
    grasp_poses = []
    with open(tmp_dir + 'gpd_grasp_poses.txt') as f:
        lines = f.readlines()
        for idx in range(0, len(lines), 3):
            grasp_pose = []
            for i in range(3):
                nums = lines[idx + i].split()
                grasp_pose.append([float(n) for n in nums])
            grasp_poses.append(grasp_pose)

    return np.array(grasp_poses)


def coord_to_transform(grasp_pose):
    quat = grasp_pose[2]

    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom - 0.03 * getApproach

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True)
    transf2 = R.from_euler('z', -90, degrees=True)
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))


def approach_coord_to_transform(grasp_pose):
    quat = grasp_pose[2]

    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom + 0.12 * getApproach

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True)
    transf2 = R.from_euler('z', -90, degrees=True)
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))

print('Starting')
rospy.init_node('listener', anonymous=True)
signal.signal(signal.SIGINT, handler)  # catch ctrl+c
point_cloud = rospy.wait_for_message("/my_gen3/rgbd_camera/depth/points", PointCloud2)

tmp_dir = '/home/haoming/Downloads/3DSGrasp-master/tmp_data/'

pc = []
for p in pc2.read_points(point_cloud, field_names=("x", "y", "z"), skip_nans=True):
    # print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
    pc.append([p[0], p[1], p[2]])

# Segmentation of Point Cloud
xyz = np.asarray(pc)
idx = np.where(xyz[:, 2] < 1.2)     # Prune point cloud to 0.6 meters from camera in z direction
xyz = xyz[idx]

print("Partial Point Cloud before Segmentation")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.visualization.draw_geometries([pcd])

plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                         ransac_n=5,
                                         num_iterations=1000)
[a, b, c, d] = plane_model

# Partial Point Cloud
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)

outlier_cloud, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=60,
                                                              std_ratio=0.6)
print("Input Partial Point Cloud")
o3d.visualization.draw_geometries([outlier_cloud])

# Save point clouds in pcd format
o3d.io.write_point_cloud(tmp_dir + 'partial_pc.pcd', outlier_cloud)
o3d.io.write_point_cloud(tmp_dir + 'original_pc.pcd', pcd)

# Save point clouds in xyz format
o3d.io.write_point_cloud(tmp_dir + 'partial_pc.xyz', outlier_cloud)
o3d.io.write_point_cloud(tmp_dir + 'original_pc.xyz', pcd)

pts_partial = np.asarray(outlier_cloud.points)
print("pts_partial: ", pts_partial.shape)
min_values = np.min(pts_partial, axis=0)
max_values = np.max(pts_partial, axis=0)


print()
print('Object completion')

local_pointnet = LocalPoolPointnet(
    c_dim=8,  # Matches the `local_pointnet.fc_c` output size in the state_dict
    dim=3,  # 3D points
    hidden_dim=32,  # Inferred from `local_pointnet.fc_pos.weight` and `local_pointnet.blocks`
    scatter_type='max',  # Assuming max pooling based on typical usage
    unet=True,  # UNet is used based on the state_dict (`local_pointnet.unet`)
    unet_kwargs={"depth": 4, "start_filts": 32, "merge_mode": "concat"},  # Example parameters, adjust as needed
    unet3d=False,  # UNet3D is not used based on the state_dict
    plane_resolution=64,  # Based on typical resolution sizes for 3D features
    grid_resolution=None,  # No grid resolution since UNet3D is not used
    plane_type=['xz', 'xy', 'yz'],  # Inferred from the typical use of planes in similar models
    padding=0.1,
    n_blocks=5  # Matches the number of blocks in `local_pointnet.blocks`
)

decoder = LocalDecoderAttn(
    dim=3,  # Input dimension
    c_dim=8,  # Matches the `decoder.fc_c` input size in the state_dict
    hidden_size=32,  # Inferred from `decoder.fc_p.weight` and `decoder.blocks`
    n_blocks=5,  # Number of blocks in the decoder
)

vae = AutoencoderKLCLI(
    in_channels=24,
    out_channels=24,
    latent_channels=12,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels=[64, 128, 256, 256],
    norm_num_groups=32,
    layers_per_block=2,
    act_fn="silu"
)

unet = UNet2DModelCLI(
    in_channels=24,
    out_channels=12,
    attention_head_dim=8,
    block_out_channels=[224, 448, 672, 896],
    down_block_types=["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
    up_block_types=["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
    layers_per_block=2,
    downsample_padding=1,
    flip_sin_to_cos=True,
    freq_shift=0,
    mid_block_scale_factor=1,
    norm_eps=1e-05,
    norm_num_groups=32,
    center_input_sample=False
)

scheduler = PNDMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1,
    beta_schedule="scaled_linear",
    skip_prk_steps=True,
    set_alpha_to_one=False,
    trained_betas=None
)

model = Diffusion3XC(
    decoder=decoder,
    local_pointnet=local_pointnet,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    feat_dim=8,  # Matches `local_pointnet.fc_c` output dimension in state_dict
    vox_reso=64,  # Based on typical 3D feature resolutions
    latent_dim=12,  # Matches latent dimension in `vae.encoder`
    plane_types=['xy', 'xz', 'yz']  # Inferred from typical usage and matches LocalDecoderAttn
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


pts_partial = farthest_point_sample(pts_partial,2048)
pts_partial = pts_partial[:,:3]
centroid = np.mean(pts_partial, axis=0)
pc = pts_partial - centroid
m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
pc = pc / m
pts_partial = pc.reshape(2048, 3)

min_coords = np.min(pts_partial, axis=0)
max_coords = np.max(pts_partial, axis=0)
print("min_coords: ", min_coords)
print("max_coords: ", max_coords)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = pts_partial[:, 0]
y = pts_partial[:, 1]
z = pts_partial[:, 2]
ax.scatter(x, y, z, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


pts_partial = torch.tensor(pts_partial, dtype=torch.float32).unsqueeze(0)
pts_partial = pts_partial.to(device)
pts_partial = pts_partial.reshape(1, 2048, 3)


checkpoint = torch.load('/home/haoming/Downloads/3DSGrasp-master/Completion/last.ckpt', map_location=device)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict, strict=True)
print("Checkpoint Loaded Successfully!")

model.eval()
with torch.no_grad():
    mesh_list = model.generate_mesh(pts_partial, num_samples=1)

print("Predicted Mesh")
print(mesh_list)
mesh = mesh_list[0][0]
mesh.show()
mesh.export('/home/haoming/Downloads/3DSGrasp-master/tmp_data/generated_mesh.obj')

# Sample points from the mesh
sampled_points, face_indices = mesh.sample(8192, return_index=True)

complete_pc = np.asarray(sampled_points)

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(sampled_points))
print("Sampled Points on the Surface of the Mesh")
o3d.visualization.draw_geometries([pcd])

pcd_r = complete_pc * (m + (m / 6))
pcd_r = pcd_r + centroid
pcdd = o3d.geometry.PointCloud()
pcdd.points = o3d.utility.Vector3dVector(pcd_r.squeeze())

o3d.io.write_point_cloud('/home/haoming/Downloads/3DSGrasp-master/tmp_data/complete_pc_vae.pcd',pcdd)
np.savetxt('outputfile.xyz', complete_pc)


print('Complete grasps')
subprocess.call('./detect_grasps /home/haoming/Downloads/3DSGrasp-master/gpd/cfg/eigen_params.cfg /home/haoming/Downloads/3DSGrasp-master/tmp_data/complete_pc.pcd',
                shell=True, cwd='/home/haoming/Downloads/3DSGrasp-master/gpd/build')

grasp_poses = read_grasps()
print('Number of grasp poses: ', grasp_poses.shape[0])
#######  ---------------------------// ---------------------------------- #####
# change grasp [k]
# pose = coord_to_transform(grasp_poses[0])
k = grasp_poses[0]
pose = coord_to_transform(k)
approach = approach_coord_to_transform(k)

#######  ---------------------------// ---------------------------------- #####

print('Pose')
print(pose)

# rospy.init_node('my_static_tf2_broadcaster')

my_pose = Pose()
my_pose.position.x = pose[0]
my_pose.position.y = pose[1]
my_pose.position.z = pose[2]
my_pose.orientation.x = pose[3]
my_pose.orientation.y = pose[4]
my_pose.orientation.z = pose[5]
my_pose.orientation.w = pose[6]

# Transform the pose from the camera frame to the base frame (world)
transformed_pose = transform_pose(my_pose, "camera_depth_frame", "base_link")

print(transformed_pose)

final_pose = np.array([transformed_pose.position.x,
                       transformed_pose.position.y,
                       transformed_pose.position.z,
                       transformed_pose.orientation.x,
                       transformed_pose.orientation.y,
                       transformed_pose.orientation.z,
                       transformed_pose.orientation.w])

# Publish the transform as goal_frame topic for visualization in RViz
publish_tf(final_pose[0], final_pose[1], final_pose[2], final_pose[3], final_pose[4], final_pose[5], final_pose[6],
           child_frame='complete_goal_frame')

np.save(tmp_dir + 'final_pose.npy', final_pose)

### Final approach
my_pose = Pose()
my_pose.position.x = approach[0]
my_pose.position.y = approach[1]
my_pose.position.z = approach[2]
my_pose.orientation.x = approach[3]
my_pose.orientation.y = approach[4]
my_pose.orientation.z = approach[5]
my_pose.orientation.w = approach[6]

rospy.init_node('listener', anonymous=True)
### Transform the pose from the camera frame to the base frame (world)
transformed_pose = transform_pose(my_pose, "camera_depth_frame", "base_link")

print(transformed_pose)

final_pose = np.array([transformed_pose.position.x,
                       transformed_pose.position.y,
                       transformed_pose.position.z,
                       transformed_pose.orientation.x,
                       transformed_pose.orientation.y,
                       transformed_pose.orientation.z,
                       transformed_pose.orientation.w])

np.save(tmp_dir + 'final_approach.npy', final_pose)
print("############## Final Approach Saved #############")