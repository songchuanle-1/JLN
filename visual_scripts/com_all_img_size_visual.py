import json
import matplotlib.pyplot as plt

# 定义 JSON 文件路径
json_file_path = '/sdc1/songcl/mono3D/visual_project/other_json/all_img_size_sacle.json'

try:
    # 打开并读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取所有的值到列表中
    values = list(data.values())

    # 对值列表进行排序
    sorted_values = sorted(values,reverse=True)

    # 计算值列表的平均值
    average_value = sum(sorted_values) / len(sorted_values)

    # 绘制曲线图
    plt.plot(sorted_values,color='pink')

    plt.xlabel('Sample')
    plt.ylabel('Percentage')
    # plt.title('Sorted Values Plot')

    # 绘制表示平均值的横虚线
    plt.axhline(y=average_value, color='r', linestyle='--', label=f'Average: {average_value:.3f}')
    plt.legend()
    # 获取当前的坐标轴
    ax = plt.gca()

    # 去除顶端和右边的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # # 设置 x 轴和 y 轴的箭头，调整箭头大小
    # arrow_props = dict(facecolor='black', width=0.2, headwidth=3, headlength=3, shrink=0.05)
    # ax.annotate('', xy=(1, 0), xycoords='axes fraction', xytext=(0.98, 0), arrowprops=arrow_props)
    # ax.annotate('', xy=(0, 1), xycoords='axes fraction', xytext=(0, 0.98), arrowprops=arrow_props)

    # 显示图形
    plt.savefig('0.png',dpi=500)

except FileNotFoundError:
    print(f"未找到文件: {json_file_path}")
except json.JSONDecodeError:
    print(f"无法解析 JSON 文件: {json_file_path}")
except Exception as e:
    print(f"发生其他错误: {e}")