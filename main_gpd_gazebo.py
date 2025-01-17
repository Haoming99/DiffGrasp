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
    bottom = grasp_bottom - 0 * getApproach # 0.03

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True) #-90
    transf2 = R.from_euler('z', -90, degrees=True) #-90
    matF = transf1.apply(quat_to_mat)
    matF = transf2.apply(matF)
    matF_to_quat = R.from_matrix(matF).as_quat()

    return np.concatenate((bottom, matF_to_quat))


def approach_coord_to_transform(grasp_pose):
    quat = grasp_pose[2]

    getApproach = np.array(grasp_pose[1])
    grasp_bottom = grasp_pose[0]
    bottom = grasp_bottom + 0 * getApproach # 0.12

    quatR = R.from_quat(quat)
    quat_to_mat = quatR.as_matrix()

    transf1 = R.from_euler('y', -90, degrees=True) #-90
    transf2 = R.from_euler('z', -90, degrees=True) #-90
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

print()
print('Object completion')
subprocess.run(
    ["/home/haoming/anaconda3/envs/3dsg_venv/bin/python3.9", "/home/haoming/Downloads/3DSGrasp-master/Completion/main.py", "--test",
     "--ckpts", "/home/haoming/Downloads/3DSGrasp-master/Completion/3dsgrasp_model.pth", "--config",
     "/home/haoming/Downloads/3DSGrasp-master/Completion/cfgs/YCB_models/SGrasp.yaml"])

compl_pc = o3d.io.read_point_cloud('/home/haoming/Downloads/3DSGrasp-master/tmp_data/complete_pc_x.pcd')

print("Complete Point Cloud")
o3d.visualization.draw_geometries([compl_pc, outlier_cloud])

merged_xyz = np.concatenate((np.asarray(compl_pc.points), np.asarray(pcd.points)), axis=0)
merged_pc = o3d.geometry.PointCloud()
merged_pc.points = o3d.utility.Vector3dVector(merged_xyz)
o3d.visualization.draw_geometries([merged_pc])

pub = rospy.Publisher("/my_gen3/rgbd_camera/depth/image_raw", PointCloud2, queue_size=2)

points = []
for i in range(merged_xyz.shape[0]):
    points.append([merged_xyz[i, 0] + 0.02, merged_xyz[i, 1], merged_xyz[i, 2]])

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          # PointField('rgba', 12, PointField.UINT32, 1),
          ]

header = Header()
header.frame_id = "camera_depth_frame"
pc2 = point_cloud2.create_cloud(header, fields, points)

for i in range(100):
    pc2.header.stamp = rospy.Time.now()
    pub.publish(pc2)
    rospy.sleep(0.1)

# exit(0)
print()
print('Partial grasps')

subprocess.call('./detect_grasps /home/haoming/Downloads/3DSGrasp-master/gpd/cfg/eigen_params.cfg /home/haoming/Downloads/3DSGrasp-master/tmp_data/partial_pc.pcd',
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
           child_frame='partial_goal_frame')

np.save(tmp_dir + 'final_pose.npy', final_pose)

print()
print('Complete grasps')
subprocess.call('./detect_grasps /home/haoming/Downloads/3DSGrasp-master/gpd/cfg/eigen_params.cfg /home/haoming/Downloads/3DSGrasp-master/tmp_data/complete_pc.pcd',
                shell=True, cwd='/home/haoming/Downloads/3DSGrasp-master/gpd/build')

### Take the first grasp proposal from GPD
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