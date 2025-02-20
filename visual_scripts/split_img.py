import os
import json
import numpy as np
import cv2

img_folder='/sdc1/songcl/mono3D/visual_project/featuremp_visual2/visual'
save_img_folder=img_folder+'_split'
if not os.path.exists(save_img_folder):os.mkdir(save_img_folder)
# classes=['','','','',]
split_json_path='/sdc1/songcl/mono3D/data/mono3DRefer/test_instanceID_split.json'
with open(split_json_path, 'r', encoding='utf-8') as json_file:
    split_json = json.load(json_file)
classes=list(split_json.keys())
save_img_split_folder=[]
for i in range(len(classes)):
    save_img_split_folder_one =os.path.join(save_img_folder,classes[i])
    if not os.path.exists(save_img_split_folder_one):os.mkdir(save_img_split_folder_one)
    save_img_split_folder.append(save_img_split_folder_one)
imglists=os.listdir(img_folder)
for i in range(len(imglists)):
    img_name=imglists[i]
    img=cv2.imread(os.path.join(img_folder,img_name))

    instanceid=int(img_name.split('_')[1])
    #区分unique
    if instanceid in split_json['Unique']:
        print(f'{img_name} Unique')
        cv2.imwrite(os.path.join(save_img_split_folder[0],img_name),img)
    else:
        print(f'{img_name} Multiple')
        cv2.imwrite(os.path.join(save_img_split_folder[1],img_name),img)

    ## 区分Far near
    if instanceid in split_json['Near']:
        print(f'{img_name} Near')
        cv2.imwrite(os.path.join(save_img_split_folder[2],img_name),img)
    elif instanceid in split_json['Medium']:
        print(f'{img_name} Medium')
        cv2.imwrite(os.path.join(save_img_split_folder[3],img_name),img)
    elif instanceid in split_json['Far']:
        print(f'{img_name} Far')
        cv2.imwrite(os.path.join(save_img_split_folder[4],img_name),img)
    ## 区分easy
    if instanceid in split_json['Easy']:
        print(f'{img_name} Easy')
        cv2.imwrite(os.path.join(save_img_split_folder[5],img_name),img)
    elif instanceid in split_json['Moderate']:
        print(f'{img_name} Moderate')
        cv2.imwrite(os.path.join(save_img_split_folder[6],img_name),img)
    elif instanceid in split_json['Hard']:
        print(f'{img_name} Hard')
        cv2.imwrite(os.path.join(save_img_split_folder[7],img_name),img)
