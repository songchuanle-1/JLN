import json
import os
import cv2
import numpy as np
Mono3DRefer_another_version_path = '/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer_another_version.json'
save_jk_path = '/sdc1/songcl/mono3D/visual_project/other_json/all_img_size_sacle.json'

img_folder = '/sdc1/songcl/mono3D/data/mono3DRefer/images'
with open(Mono3DRefer_another_version_path, 'r', encoding='utf-8') as json_file:
    Mono3DRefer_another_version_json = json.load(json_file)

jk = list(Mono3DRefer_another_version_json.keys())

save_json = {}
for i in range(len(jk)):
    jk_name = jk[i]

    img_name = jk_name.split('_')[0]
    img_path = os.path.join(img_folder,img_name+'.png')
    img = cv2.imread(img_path)
    h,w,c = img.shape
    gt_label_2 = Mono3DRefer_another_version_json[jk_name]['label_2']
    gt_label_2 = np.array([float(x) for x in gt_label_2[1:-1].split(',')[1:]])

    xyxy = gt_label_2[3:7]
    sub_x = xyxy[2]-xyxy[0]
    sub_y = xyxy[3]-xyxy[1]
    area = sub_x*sub_y
    img_area = h*w
    save_json[jk_name] = area/img_area
    print(f'{jk_name} res {area/img_area}')


try:
    # 打开文件以写入模式
    with open(save_jk_path, 'w') as json_file:
        # 使用 json.dump 将列表写入文件
        json.dump(save_json, json_file,indent=2)
    print(f"列表已成功保存到 {save_json}")
except Exception as e:
    print(f"保存列表到文件时出现错误: {e}")

    # print(xyxy)
    # x1 = int(xyxy[0])
    # y1 = int(xyxy[1])
    # x2 = int(xyxy[2])
    # y2 = int(xyxy[3])
    # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2,2)
    # cv2.imwrite('0.png',img)


    # pp
