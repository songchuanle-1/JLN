import os
import json
import cv2
import matplotlib.pyplot as plt
# from visual import read_calib_file, project_3d_to_2d,compute_box_3d
import numpy as np
def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as json_file:
        # 使用 json.load 解析文件中的 JSON 数据
        all_keys =[]
        all_values = []
        my_list = json.load(json_file)
        for key, value in my_list.items():
            all_keys.append(key)
            all_values.append(value)
        ## 读取所有的key,value
        return all_keys, all_values
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

def visual_img(gt_dim,gt_location,gt_rotation_y,dim,location,rotation_y,calib,image_path):
        # 计算 3D 框的 8 个顶点
    P = calib['P2']
    gt_corners_3d = compute_box_3d(gt_dim,gt_location,gt_rotation_y)
    gt_corners_2d = project_3d_to_2d(gt_corners_3d, P)  # 投影 3D 点到图像平面
    corners_3d = compute_box_3d(dim,location,rotation_y)
    corners_2d = project_3d_to_2d(corners_3d, P)  # 投影 3D 点到图像平面
    # 绘制 2D 框
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    col = [(0, 255, 0),(0, 0, 255)]
    
    image = cv2.imread(image_path)
    for edge in edges:
        cv2.line(image, tuple(gt_corners_2d[edge[0]]), tuple(gt_corners_2d[edge[1]]), col[0] , 2)
    for edge in edges:
        cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), col[1] , 2)

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
    # plt.savefig(save_path_3d)
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

    return combined_image






## 读取官方的json文件 gt
Mono3DRefer_another_version_path = '/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer_another_version.json'
with open(Mono3DRefer_another_version_path, 'r', encoding='utf-8') as json_file:
    Mono3DRefer_another_version_json = json.load(json_file)

## 读取baseline 和JLN的json文件  方便进行比较
baseline_result_json_file = '/sdc1/songcl/mono3D/Mono3DVG/output_paper_select/author_checkpoints/result.json'
JLN_result_json_file = '/sdc1/songcl/mono3D/Mono3DVG/output_paper_select/mono3dvg_add_split_loss_l2_add_l2_add_depth_add_contrastive8_3_add_injection_all4/result.json'
data_root_path = '/sdc1/songcl/mono3D/data/mono3DRefer'
save_root_path = '/sdc1/songcl/mono3D/visual_project/compare'
if not os.path.exists(save_root_path):os.mkdir(save_root_path)
baseline_save_root_path = save_root_path+'/baseline'
JLN_save_root_path = save_root_path+'/JLN'
all_save_root_path = save_root_path+'/All'
if not os.path.exists(baseline_save_root_path):os.mkdir(baseline_save_root_path)
if not os.path.exists(JLN_save_root_path):os.mkdir(JLN_save_root_path)
if not os.path.exists(all_save_root_path):os.mkdir(all_save_root_path)


baseline_key, baseline_res_list = read_json(baseline_result_json_file)
JLN_key, JLN_res_list = read_json(JLN_result_json_file)
print('len(baseline_res_list)',len(baseline_res_list))
print('len(JLN_res_list)',len(JLN_res_list))
# pp
assert len(baseline_res_list) == len(JLN_res_list)

## 开始遍历文件
for i,(baseline_one_res,JLN_one_res) in enumerate(zip(baseline_res_list,JLN_res_list)):
    
    ### query gt boxes info
    json_key = JLN_key[i]
    gt_info = Mono3DRefer_another_version_json.get(json_key,None)
    if gt_info is not None:
        gt_label_2 = gt_info['label_2']

    ####读取信息
    baseline_im_name = baseline_one_res['im_name']
    baseline_instanceID = baseline_one_res['instanceID']
    baseline_ann_id = baseline_one_res['ann_id']
    baseline_label_2_pre = baseline_one_res['label_2_pre']
    baseline_label_2_gt = baseline_one_res['label_2_gt']
    baseline_IoU= baseline_one_res['IoU']
    baseline_description= baseline_one_res['description']

    JLN_im_name = JLN_one_res['im_name']
    JLN_instanceID = JLN_one_res['instanceID']
    JLN_ann_id = JLN_one_res['ann_id']
    JLN_label_2_pre = JLN_one_res['label_2_pre']
    JLN_label_2_gt = JLN_one_res['label_2_gt']
    JLN_IoU= JLN_one_res['IoU']
    JLN_description= JLN_one_res['description']

    assert baseline_im_name == JLN_im_name
    assert baseline_instanceID == JLN_instanceID
    assert baseline_ann_id == JLN_ann_id
    # print(f'{JLN_key[i]}')



    ## 比较IOU大小
    if baseline_IoU < JLN_IoU:
        print(f'{JLN_key[i]}_baseline_IoU:{baseline_IoU}<JLN_IoU:{JLN_IoU}')
        ## 可视化gt和pred
        ## 读取calib文件
        calib_path= data_root_path+'/calib/'+JLN_im_name+'.txt'
        
        calib = read_calib_file(calib_path)
        img_path= data_root_path+'/images/'+JLN_im_name+'.png'
        save_path = save_root_path+'/images/'+JLN_key[i]+'.png'
        if not os.path.exists(calib_path):
            print(f'path {calib_path}not exists')
            break
        if not os.path.exists(img_path):
            print(f'path {img_path} not exists')
            break
        gt_label_2 = np.array([float(x) for x in gt_label_2[1:-1].split(',')[1:]])
        baseline_pred_label_2 = np.array([float(x) for x in baseline_label_2_pre[1:-1].split(',')[1:]])
        JLN_pred_label_2 = np.array([float(x) for x in JLN_label_2_pre[1:-1].split(',')[1:]])
        # print(gt_label_2)
        # print(baseline_label_2_pre)
        # print(JLN_label_2_pre)
        gt_dim = gt_label_2[7:10]
        gt_location = gt_label_2[10:13]
        gt_rotation_y = gt_label_2[-1]
        
        baseline_dim = baseline_pred_label_2[5:8]
        baseline_location = baseline_pred_label_2[8:11]
        baseline_rotation_y = baseline_pred_label_2[-2]

        JLN_dim = JLN_pred_label_2[5:8]
        JLN_location = JLN_pred_label_2[8:11]
        JLN_rotation_y = JLN_pred_label_2[-2]
        # print(gt_dim,baseline_dim,JLN_dim)
        # print(gt_location,baseline_location,JLN_location)
        # print(gt_rotation_y,baseline_rotation_y,JLN_rotation_y)

        baseline_img = visual_img(gt_dim,gt_location,gt_rotation_y,baseline_dim,baseline_location,baseline_rotation_y,calib,img_path)
        JLN_img = visual_img(gt_dim,gt_location,gt_rotation_y,JLN_dim,JLN_location,JLN_rotation_y,calib,img_path)
        ttt = f'B:{round(baseline_IoU,3)}<J:{round(JLN_IoU,3)}'
        # cv2.imwrite(os.path.join(baseline_save_root_path,f'{json_key}_{ttt}.png'),baseline_img)
        # cv2.imwrite(os.path.join(JLN_save_root_path,f'{json_key}_{ttt}.png'),JLN_img)

        v_image = np.vstack((baseline_img,JLN_img))
        cv2.imwrite(os.path.join(all_save_root_path,f'{json_key}_{ttt}.png'),v_image)












    
