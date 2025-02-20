import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 读取校准文件
def read_calib_file(file_path):
    calib = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            # print(len(line))
            if len(line)==1:continue
            key = line[0][:-1]
            # print(key)
            values = [float(x) for x in line[1:]]
            if key == 'R0_rect':
                # R0_rect 是 3x3 矩阵
                value = np.array(values).reshape((3, 3))
            else:
                # 其他参数是 3x4 矩阵
                value = np.array(values).reshape((3, 4))
            calib[key] = value
    return calib

# 从 3D 框信息生成 3D 框的 8 个顶点
def compute_box_3d(dim, location, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += location[0]
    corners_3d[1, :] += location[1]
    corners_3d[2, :] += location[2]
    return corners_3d

# 将 3D 点投影到图像平面
def project_3d_to_2d(points_3d, P):
    points_3d_homogeneous = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
    points_2d_homogeneous = np.dot(P, points_3d_homogeneous)
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    return points_2d.astype(np.int32).T


# 设置 3D 坐标轴尺度一致
def set_axes_equal(ax):
    """设置 3D 坐标轴尺度一致"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # 找出最大范围
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# 读取校准文件
# calib_path= '/sdc1/songcl/mono3D/data/mono3DRefer/calib/000006.txt'
# calib = read_calib_file(calib_path)
# 读取 JSON 文件
with open('/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer.json', 'r') as f:
    datas = json.load(f)

for i in range(len(datas)):
    
    # if datas[i]['im_name']=='001714' :
    #     # and  datas[i]['ann_id']==2
    # #     continue
    # # else:
    data = datas[i]
    print(str(data['instanceID'])+'_'+data['im_name']+'_'+data['objectName']+'_'+str(datas[i]['ann_id']))
    calib_path= '/sdc1/songcl/mono3D/data/mono3DRefer/calib/'+data['im_name']+'.txt'
    # import os
    # print(os.path.exists(calib_path))
    calib = read_calib_file(calib_path)
    # 读取图像
    image_path = calib_path
    image_path = image_path.replace("calib", "images")
    image_path = image_path.replace(".txt", ".png")
    save_path_root = '/sdc1/songcl/mono3D/visual_project/framework_page/'
    ip = str(data['instanceID'])+'_'+data['im_name']+'_'+data['objectName']+'_'
    import os
    if not os.path.exists(save_path_root):os.mkdir(save_path_root)
    save_path = save_path_root+ip+str(datas[i]['ann_id'])+'.png'
    save_path_concat = save_path_root+ip+str(datas[i]['ann_id'])+'_concat.png'
    save_path_3d = save_path_root+ip+str(datas[i]['ann_id'])+'_3d.png'
    # 提取 3D 框信息
    box_3d = np.array([float(x) for x in data['Box_3D'][1:-1].split(',')])
    label_2 = np.array([float(x) for x in data['label_2'][1:-1].split(',')[1:]])
    # location = box_3d[:3]
    # print(label_2)
    dim = label_2[7:10]
    location = label_2[10:13]
    # print(location)
    # print(dim)


    rotation_y = float(data['label_2'].split(',')[-1].strip().replace(']', ''))

    # 计算 3D 框的 8 个顶点
    corners_3d = compute_box_3d(dim, location, rotation_y)

    # 投影 3D 点到图像平面
    P = calib['P2']
    corners_2d = project_3d_to_2d(corners_3d, P)


    image = cv2.imread(image_path)

    # 绘制 2D 框
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), (0, 255, 0), 2)

    # 显示图像
    cv2.imwrite(save_path, image)


    # 绘制 3D 框
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制顶点
    ax.scatter(corners_3d[0, :], corners_3d[2, :], corners_3d[1, :], c='r', marker='o')

    # 绘制边
    for edge in edges:
        ax.plot([corners_3d[0, edge[0]], corners_3d[0, edge[1]]],
                [corners_3d[2, edge[0]], corners_3d[2, edge[1]]],
                [corners_3d[1, edge[0]], corners_3d[1, edge[1]]], c='g')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    # 显示 3D 图
    plt.savefig(save_path_3d)


    # 将 plt 绘制的图像转换为 numpy 数组
    fig = plt.gcf()
    fig.canvas.draw()
    image_from_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plt = image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 调整 plt 图像的高度与 cv2 图像一致
    if image_from_plt.shape[0] != image.shape[0]:
        new_width = int(image_from_plt.shape[1] * (image.shape[0] / image_from_plt.shape[0]))
        image_from_plt = cv2.resize(image_from_plt, (new_width, image.shape[0]))

    # 拼接图像
    combined_image = np.hstack((image, image_from_plt))

    cv2.imwrite(save_path_concat, combined_image)