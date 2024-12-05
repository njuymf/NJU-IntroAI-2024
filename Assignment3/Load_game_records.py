import glob
import os
import pickle
import numpy as np


def load_game_records(date_prefix='2024-11-22'):
    # 获取当前脚本的目录
    current_dir = os.path.dirname(__file__)
    print(f"当前工作目录: {current_dir}")
    # 返回上一级目录并拼接到logs
    logs_dir = os.path.join(current_dir, 'logs')
    print(f"游戏记录目录: {logs_dir}")

    # 在上一级目录中查找
    data_list = glob.glob(os.path.join(logs_dir, f'game_records_lvl1_{date_prefix}*'))
    all_data = []

    print(f"正在搜索日期前缀为 {date_prefix} 的游戏记录...")
    for data_path in data_list:
        data_file = os.path.join(data_path, 'data.pkl')
        try:
            with open(data_file, 'rb') as f:
                loaded_data = pickle.load(f)
                all_data.extend(loaded_data)
                print(f"从 {data_file} 加载了 {len(loaded_data)} 条记录")
        except Exception as e:
            print(f"加载 {data_file} 时出错: {e}")

    return all_data


def inspect_data(data):  
    if not data:  
        print("数据为空，无法进行检查")  
        return  

    print(f"数据集中包含的示例数量: {len(data)}")  
    
    for i, (observation, action) in enumerate(data[:5]):  # 只查看前5个样本  
        print(f"示例 {i + 1}:")  
        print(f"  观察 (observation) 类型: {type(observation)}")  
        print(f"  动作 (action) 类型: {type(action)}")  
        print(f"  观察样本数据: {observation}")  # 打印观察的数据  
        try:  
            print(f"  观察形状: {np.array(observation).shape}")  # 尝试输出形状  
        except ValueError as e:  
            print(f"  观察形状获取错误: {e}")  # 捕捉并打印异常  
        print(f"  动作: {action}")  



if __name__ == '__main__':
    data = load_game_records()
    print("数据集示例:")
    print(data[200])
    print("数据集大小：")
    print(len(data))
    print("单个数据规格:")
    print(len(data[0]))
    print("列大小:")
    print(len(data[0][0]))
    print("行大小:")
    print(len(data[0][0][0]))