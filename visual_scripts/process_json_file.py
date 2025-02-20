import os
import json
def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as json_file:
        # 使用 json.load 解析文件中的 JSON 数据
        my_list = json.load(json_file)
        return my_list
json_file = '/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer_copy.json'
save_path = '/sdc1/songcl/mono3D/data/mono3DRefer/Mono3DRefer_another_version.json'
json_list = read_json(json_file)
json_dicts = {}
for i in range(len(json_list)):
    list_one = json_list[i]
    list_one_name = str(list_one['im_name']) +'_'+ str(list_one['instanceID'])+'_'+ str(list_one['ann_id'])
    print(list_one_name)
    if json_dicts.get(list_one_name,None) is not None:
        print(f'key:{list_one_name} retry')
        continue
    else:
        json_dicts[list_one_name] = list_one

file_path = os.path.join(save_path)

try:
    # 打开文件以写入模式
    with open(file_path, 'w') as json_file:
        # 使用 json.dump 将列表写入文件
        json.dump(json_dicts, json_file,indent=2)
    print(f"列表已成功保存到 {file_path}")
except Exception as e:
    print(f"保存列表到文件时出现错误: {e}")