import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def plot_smpl(pose, inplace=True, return_img=False):
    """Plot the 3D pose showing the joint connections.
    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', #5
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', #10
    'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', #15
    'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', #20
    'R_Wrist', 'L_Hand', 'R_Hand']
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

    _CONNECTION = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                                    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13],
                                    [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
                                    [18, 20], [19, 21], [20, 22], [21, 23]])

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [23, 21, 19, 17, 14]:
            _c = 1
        if j in [2, 5, 8, 11]:
            _c = 2
        if j in [13, 16, 18, 20, 22]:
            _c = 3
        if j in [1, 4, 7, 10]:
            _c = 4
        # if j in range ( 14, 17 ):
        #     _c = 5
        return colors[_c]

    fig = plt.figure ()
    if return_img:
        canvas = FigureCanvas(fig)
    if inplace:
        ax = plt.axes(projection='3d')
        ax.view_init ( -120, 90 )  # To adjust weird viewpoint
    import math

    pose = np.array(pose).T
    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    if not inplace:
        ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color ( c[1] )
        if np.any(np.isnan(pose[:, c[0]])) or np.any(np.isnan(pose[:, c[0]])):
            continue #Ignore nan points
        ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                  [pose[1, c[0]], pose[1, c[1]]],
                  [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle='-' )

    for j in range ( pose.shape[1] ):
        col = '#%02x%02x%02x' % joint_color ( j )

        if np.any(np.isnan(pose[:, j])):
            continue
        ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                     c=col, marker='x' )
    smallest = np.nanmin(pose)
    largest = np.nanmax(pose)
    ax.set_xlim3d ( smallest, largest )
    ax.set_ylim3d ( smallest, largest )
    ax.set_zlim3d ( smallest, largest )

    if return_img:
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(fig)
        return image
    else:
        return fig

def plot_multi_kpts(poses, inplace=True, return_img=False):
    """Plot the 3D pose showing the joint connections.
    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', #5
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', #10
    'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', #15
    'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', #20
    'R_Wrist', 'L_Hand', 'R_Hand']
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

    _CONNECTION = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                                    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13],
                                    [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
                                    [18, 20], [19, 21], [20, 22], [21, 23]])

    colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]

    fig = plt.figure ()
    if return_img:
        canvas = FigureCanvas(fig)
    if inplace:
        ax = plt.axes(projection='3d')
        ax.view_init ( -120, 90 )  # To adjust weird viewpoint


    smallest = np.nanmin(poses)
    largest = np.nanmax(poses)
    ax.set_xlim3d ( smallest, largest )
    ax.set_ylim3d ( smallest, largest )
    ax.set_zlim3d ( smallest, largest )

    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    for i, pose in enumerate(poses):
        pose = pose.T
        col = '#%02x%02x%02x' % colors[i % len(colors)]
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
            ax.set_xlim3d ( smallest, largest )
            ax.set_ylim3d ( smallest, largest )
            ax.set_zlim3d ( smallest, largest )

        for c in _CONNECTION:
            if np.any(np.isnan(pose[:, c[0]])) or np.any(np.isnan(pose[:, c[0]])):
                continue #Ignore nan points
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle='-' )

        for j in range ( pose.shape[1] ):

            if np.any(np.isnan(pose[:, j])):
                continue
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='x' )

    if return_img:
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(fig)
        return image
    else:
        return fig

def viz_pose_voxels_img(pose, voxels):

    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

    _CONNECTION = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                                    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13],
                                    [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
                                    [18, 20], [19, 21], [20, 22], [21, 23]])

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [23, 21, 19, 17, 14]:
            _c = 1
        if j in [2, 5, 8, 11]:
            _c = 2
        if j in [13, 16, 18, 20, 22]:
            _c = 3
        if j in [1, 4, 7, 10]:
            _c = 4
        # if j in range ( 14, 17 ):
        #     _c = 5
        return colors[_c]

    fig = plt.figure (dpi=150)
    canvas = FigureCanvas(fig)
    pose = np.array(pose).T
    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init ( 30, 90 )  # To adjust weird viewpoint
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color ( c[1] )
        if np.any(np.isnan(pose[:, c[0]])) or np.any(np.isnan(pose[:, c[0]])):
            continue #Ignore nan points
        ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                  [pose[1, c[0]], pose[1, c[1]]],
                  [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle='-' )

    for j in range ( pose.shape[1] ):
        col = '#%02x%02x%02x' % joint_color ( j )

        if np.any(np.isnan(pose[:, j])):
            continue
        ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                     c=col, marker='x' )
    smallest = np.nanmin(pose)
    largest = np.nanmax(pose)
    ax.set_xlim3d ( smallest, largest )
    ax.set_ylim3d ( smallest, largest )
    ax.set_zlim3d ( smallest, largest )

    # Use numpy
    voxels = np.asarray(voxels)
    voxels = voxels.transpose(2, 0, 1)

    # Create plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    canvas.draw()       # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.close(fig)
    return image

def plot_pose_voxels(pose, voxels, return_img=False, figsize=None, dpi=None):
    """Plot the 3D pose showing the joint connections.
    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', #5
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', #10
    'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', #15
    'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', #20
    'R_Wrist', 'L_Hand', 'R_Hand']
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    smpl_joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_shoulder', 'R_shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

    _CONNECTION = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                                    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13],
                                    [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19],
                                    [18, 20], [19, 21], [20, 22], [21, 23]])

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [23, 21, 19, 17, 14]:
            _c = 1
        if j in [2, 5, 8, 11]:
            _c = 2
        if j in [13, 16, 18, 20, 22]:
            _c = 3
        if j in [1, 4, 7, 10]:
            _c = 4
        # if j in range ( 14, 17 ):
        #     _c = 5
        return colors[_c]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if return_img:
        canvas = FigureCanvas(fig)
    import math

    pose = np.array(pose).T
    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    plt_idx = 1
    ax = fig.add_subplot(1, len(voxels)+1, plt_idx, projection='3d' )
    plt_idx += 1
    ax.view_init( -120, 90 )  # To adjust weird viewpoint

    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color ( c[1] )
        if np.any(np.isnan(pose[:, c[0]])) or np.any(np.isnan(pose[:, c[0]])):
            continue #Ignore nan points
        ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                  [pose[1, c[0]], pose[1, c[1]]],
                  [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle='-' )

    for j in range ( pose.shape[1] ):
        col = '#%02x%02x%02x' % joint_color ( j )

        if np.any(np.isnan(pose[:, j])):
            continue
        ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                     c=col, marker='x' )
    smallest = np.nanmin(pose)
    largest = np.nanmax(pose)
    ax.set_xlim3d ( smallest, largest )
    ax.set_ylim3d ( smallest, largest )
    ax.set_zlim3d ( smallest, largest )
    ax.title.set_text('gt kpts')


    for i, vox_occ in enumerate(voxels):
        ax = fig.add_subplot(1, len(voxels)+1, plt_idx, projection='3d')
        plt_idx += 1
        ax.voxels(vox_occ, edgecolor="k")
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel('z')
        ax.view_init(elev=120, azim=-90)
        if i == 0:
            ax.title.set_text('gt voxel')
        else:
            ax.title.set_text(f'sample #{i-1}')

    if return_img:
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(fig)
        return image
    else:
        return fig

def plot_pcs(pcs, b_min=-0.55, b_max=0.55, figsize=(5, 5), dpi=100, return_img=False):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if return_img:
        canvas = FigureCanvas(fig)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcs[:, 0], pcs[:, 2], pcs[:, 1])
    ax.set_xlim3d(-0.55, 0.55)
    ax.set_ylim3d(-0.55, 0.55)
    ax.set_zlim3d(-0.55, 0.55)
    if return_img:
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(fig)
        return image
    else:
        return fig
