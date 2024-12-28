# DiffGrasp Manual
This repository contains the implementation of our paper "DiffGrasp: Diffusion-based 3D Shape Completion for Robotic Grasping" . 

The codes of our shape completion method are mainly in `DiffGrasp/Completion/diff_models`.

The codes of robotic grasping experiments are mainly in `DiffGrasp/main_gpd_diff_.py` and `DiffGrasp/main_gpd_diff_gazebo.py`


## Environment setup
**Commands for creating an environment and further setup**:
```bash
        cd DiffGrasp
	conda env create -f diffgrasp.yaml
	conda activate diffgrasp
	cd Completion/extensions/chamfer_dist/
	pip install .
	cd ../..
	cd models/pointnet2_ops_lib
	pip install .
	pip install timm
	pip install tensorboardX
	pip install easydict
	pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```



## Full Pipeline for Kinova Gen3 Grasping in Real World

1. **Kortex Driver**:
    ```bash
    conda activate diffgrasp
    source catkin_workspace/devel/setup.bash
    roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85
    ```

2. **Kinova Vision Module**:
    ```bash
    conda activate diffgrasp
    source catkin_ws/devel/setup.bash
    roslaunch kinova_vision kinova_vision_rgbd.launch device:=$IP_KINOVA
    ```

3. **Point Cloud Completion and Grasp Pose Generation**:
    ```bash
    conda activate diffgrasp
    source catkin_workspace/devel/setup.bash
    cd Downloads/DiffGrasp/
    python main_gpd_diff.py
    ```

4. **Execute Actions in the Real World**:
    ```bash
    conda activate diffgrasp
    cd
    source catkin_workspace/devel/setup.bash
    roslaunch kortex_examples reach_approach_grasp_pose.launch
    ```

---

## Training and Evaluation for Baselines 

1. **Training**:
    ```bash
    conda activate diffgrasp
    cd Downloads/DiffGrasp/Completion
    python3 main.py --config ./cfgs/YCB_models/<baseline>.yaml
    ```

2. **Evaluation**:
    ```bash
    conda activate diffgrasp
    cd Downloads/DiffGrasp/Completion
    python3 main.py --test --ckpts default_model.pth --config ./cfgs/YCB_models/<baseline>.yaml
    ```

---

## Grasp Poses from GPD
```bash
conda activate diffgrasp
cd Downloads/DiffGrasp/gpd/build
./detect_grasps ../cfg/eigen_params.cfg /home/haoming/Downloads/DiffGrasp/tmp_data/complete_pc.pcd
```


## Gazebo Setup

To set up and execute the Gazebo simulation for Kinova Gen3 grasping, follow these steps:

1. **Spawn Gen3 in Gazebo**:
    ```bash
    conda activate diffgrasp
    source catkin_workspace/devel/setup.bash
    roslaunch kortex_gazebo spawn_kortex_robot.launch arm:=gen3 gripper:=robotiq_2f_85 dof:=7 vision:=true sim:=true
    ```

2. **Point Cloud Completion and Grasp Pose Generation**:
    ```bash
    conda activate diffgrasp
    source catkin_workspace/devel/setup.bash
    cd Downloads/DiffGrasp/
    python main_gpd_gazebo.py
    ```

3. **Execute Actions in Gazebo**:
    ```bash
    conda activate diffgrasp
    source catkin_workspace/devel/setup.bash
    roslaunch kortex_examples reach_approach_grasp_pose.launch
    ```

## Train and Evaluate the Diffusion Model on the YCB and ShapeNet Dataset
**Train the diffusion model**:
```bash
     python3 /Completion/diff_models/main_diffusion3xc_all.py
```

**Mesh Generation using the diffusion model**
```bash
     python3 /Completion/diff_models/main_mesh_generation.py
```

