import os
import json

Mono3DRefer_another_version_path = '/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer_another_version.json'
img_folder_path = '/sdc1/songcl/mono3D/visual_project/featuremp_visual2/visual_split/select'
save_file_path = img_folder_path+'/visual_caption.json'
with open(Mono3DRefer_another_version_path, 'r', encoding='utf-8') as json_file:
    Mono3DRefer_another_version_json = json.load(json_file)

imgfilelists = os.listdir(img_folder_path)

save_json = {}
for i in range(len(imgfilelists)):
    img_name = imgfilelists[i]
    arr = img_name.split('.')[0].split('_')
    json_key = arr[1]+'_'+arr[2]+'_'+arr[3]
    caption = Mono3DRefer_another_version_json[json_key]['description']
    save_json[json_key] = caption


# # 定义要保存的 JSON 文件路径
# save_file_path = os.path.join()

try:
    # 打开文件以写入模式
    with open(save_file_path, 'w') as json_file:
        # 使用 json.dump 将列表写入文件
        json.dump(save_json, json_file,indent=2)
    print(f"列表已成功保存到 {save_file_path}")
except Exception as e:
    print(f"保存列表到文件时出现错误: {e}")
