import os
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
# from compare_two_json import compute_box_3d,project_3d_to_2d,read_calib_file
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

## 读取官方的json文件 gt
Mono3DRefer_another_version_path = '/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer_another_version.json'
JLN_result_json_file = '/sdc1/songcl/mono3D/Mono3DVG/output_paper_select/mono3dvg_add_split_loss_l2_add_l2_add_depth_add_contrastive8_3_add_injection_all4/result.json'
data_root_path = '/sdc1/songcl/mono3D/data/mono3DRefer'
json_folder_path = '/sdc1/songcl/mono3D/visual_project/featuremp'
img_root_path = data_root_path+'/images'

save_root_path = '/sdc1/songcl/mono3D/visual_project/featuremp_visual2'
if not os.path.exists(save_root_path):os.mkdir(save_root_path)
save_root_path_visual = save_root_path+'/visual'
save_root_path_depth = save_root_path+'/depth'
save_root_path_3D = save_root_path+'/3D'



if not os.path.exists(save_root_path_visual):os.mkdir(save_root_path_visual)
if not os.path.exists(save_root_path_depth):os.mkdir(save_root_path_depth)
if not os.path.exists(save_root_path_3D):os.mkdir(save_root_path_3D)


select_img_folder = '/sdc1/songcl/mono3D/visual_project/compare/All'
img_lists = os.listdir(select_img_folder)
jk = [img_lists[i].split('_')[0]+'_'+img_lists[i].split('_')[1]+'_'+img_lists[i].split('_')[2]for i in range(len(img_lists))]




with open(Mono3DRefer_another_version_path, 'r', encoding='utf-8') as json_file:
    Mono3DRefer_another_version_json = json.load(json_file)

with open(JLN_result_json_file, 'r', encoding='utf-8') as json_file:
    JLN_json = json.load(json_file)





filelists= os.listdir(json_folder_path)
spatial_shapes = [[48,160],[24,80],[12,40],[6,20]]


# print(filelists)
for i in range(len(filelists)):

    json_name = filelists[i]
    img_path = os.path.join(img_root_path,json_name.split('_')[0]+'.png')
    img = cv2.imread(img_path)
    img_h,img_w,c = img.shape
    if not os.path.exists(img_path):
        print(f'no path {img_path}')
        break
    json_key = json_name.split('.')[0]
    if json_key not in jk:
        print(f'{json_key} not in select')
        continue
    print(f'{json_key} in select')

    gt_info = Mono3DRefer_another_version_json.get(json_key,None)
    JLN_info = JLN_json.get(json_key,None)
    calib_path= data_root_path+'/calib/'+json_name.split('_')[0]+'.txt'
    calib = read_calib_file(calib_path)


    if gt_info is not None and JLN_info is not None:
        print(json_name)
        gt_label_2 = gt_info['label_2']
        gt_label_2 = np.array([float(x) for x in gt_label_2[1:-1].split(',')[1:]])
        gt_dim = gt_label_2[7:10]
        gt_location = gt_label_2[10:13]
        gt_rotation_y = gt_label_2[-1]

        JLN_label_2_pre = JLN_info['label_2_pre']
        JLN_pred_label_2 = np.array([float(x) for x in JLN_label_2_pre[1:-1].split(',')[1:]])
        JLN_dim = JLN_pred_label_2[5:8]
        JLN_location = JLN_pred_label_2[8:11]
        JLN_rotation_y = JLN_pred_label_2[-2]

        P = calib['P2']
        gt_corners_3d = compute_box_3d(gt_dim,gt_location,gt_rotation_y)
        gt_corners_2d = project_3d_to_2d(gt_corners_3d, P)  # 投影 3D 点到图像平面
        corners_3d = compute_box_3d(JLN_dim,JLN_location,JLN_rotation_y)
        corners_2d = project_3d_to_2d(corners_3d, P)  # 投影 3D 点到图像平面
        # 绘制 2D 框
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        col = [(0, 255, 0),(0, 0, 255)]
        
        # image = cv2.imread(img_path)
        for edge in edges:
            cv2.line(img, tuple(gt_corners_2d[edge[0]]), tuple(gt_corners_2d[edge[1]]), col[0] , 2)
        for edge in edges:
            cv2.line(img, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), col[1] , 2)

        # 绘制 3D 框
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制顶点
        ax.scatter(gt_corners_3d[0, :], gt_corners_3d[2, :], gt_corners_3d[1, :], c='g', marker='o')

        # 绘制边
        for edge in edges:
            ax.plot([gt_corners_3d[0, edge[0]], gt_corners_3d[0, edge[1]]],
                    [gt_corners_3d[2, edge[0]], gt_corners_3d[2, edge[1]]],
                    [gt_corners_3d[1, edge[0]], gt_corners_3d[1, edge[1]]], c='g')
        # 绘制顶点
        ax.scatter(corners_3d[0, :], corners_3d[2, :], corners_3d[1, :], c='b', marker='o')

        # 绘制边
        for edge in edges:
            ax.plot([corners_3d[0, edge[0]], corners_3d[0, edge[1]]],
                    [corners_3d[2, edge[0]], corners_3d[2, edge[1]]],
                    [corners_3d[1, edge[0]], corners_3d[1, edge[1]]], c='b')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        # 显示 3D 图
        # 
        img_cp = img.copy()
        json_file_path = os.path.join(json_folder_path, json_name)
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            json_info = json.load(json_file)
        visual_list = torch.tensor(json_info['visual'])
        depth_list = torch.tensor(json_info['depth'])
        
        vl = visual_list.split([H_ * W_ for H_, W_ in spatial_shapes], dim=0)
        # for i in range(len(spatial_shapes)):
        one_view = np.array(vl[0].view(spatial_shapes[0][0],spatial_shapes[0][1]))*255
        one_view_resize = cv2.resize(one_view,(img_w,img_h))
        three_channel_weight_map = cv2.cvtColor(one_view_resize, cv2.COLOR_GRAY2BGR)
        
        # img = np.vstack((img, three_channel_weight_map))
        
        one_depth_view = np.array(depth_list.view(spatial_shapes[1][0],spatial_shapes[1][1]))*255
        one_view_resize_dpeth = cv2.resize(one_depth_view,(img_w,img_h))
        visual_img_path = os.path.join(save_root_path_visual,f'{json_key}.png')
        depth_img_path = os.path.join(save_root_path_depth,f'{json_key}.png')
        img_path_3D = os.path.join(save_root_path_3D,f'{json_key}.png')

        three_channel_weight_map_depth = cv2.cvtColor(one_view_resize_dpeth, cv2.COLOR_GRAY2BGR)
        img  = 0.2*img+0.8*three_channel_weight_map
        img_cp  = 0.2*img_cp+0.8*three_channel_weight_map_depth
        # img_cp = np.vstack((img_cp, three_channel_weight_map_depth))


        fig = plt.gcf()
        fig.canvas.draw()
        image_from_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plt = image_from_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # 调整 plt 图像的高度与 cv2 图像一致
        if image_from_plt.shape[0] != img_cp.shape[0]:
            new_width = int(image_from_plt.shape[1] * (img_cp.shape[0] / image_from_plt.shape[0]))
            image_from_plt = cv2.resize(image_from_plt, (new_width, img_cp.shape[0]))

        # 拼接图像
        img = np.hstack((img, image_from_plt))
        img_cp= np.hstack((img_cp, image_from_plt))
        cv2.imwrite(visual_img_path, img)
        cv2.imwrite(depth_img_path, img_cp)
        # plt.savefig(img_path_3D)
        # pp



    else:
        print('no json')
        continue




        


    

    
