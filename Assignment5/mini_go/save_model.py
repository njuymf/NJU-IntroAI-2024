import os  
import datetime  
import torch  

def save_model(model, iteration, filename_prefix="value_network", save_dir="parameters3"):  
    # 确保目录存在  
    os.makedirs(save_dir, exist_ok=True)  
    # 生成文件名  
    filename = f"{filename_prefix}_iter_{iteration}.pth"  
    # 构造完整的文件路径  
    file_path = os.path.join(save_dir, filename)  
    # 保存模型  
    torch.save(model.state_dict(), file_path)