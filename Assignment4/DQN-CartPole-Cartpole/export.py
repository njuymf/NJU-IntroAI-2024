import csv  
from datetime import datetime

# 在train函数前添加  
def create_csv_file(filename):  
    with open(filename, 'w', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(['Episode', 'Steps', 'Training_Loss', 'Evaluation_Return'])  

def append_to_csv(filename, episode, steps, loss, eval_return):  
    with open(filename, 'a', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow([episode, steps, loss, eval_return])  
        
def generate_filename(args):  
    """  
    根据超参数生成唯一的文件名，并放在datalog文件夹中  
    """  
    
    filename = f"datalog/{args.agent_name}_" \
               f"eps{args.epsilon_start:.2f}_" \
               f"decay{args.epsilon_decay_rate:.2f}_" \
               f"gamma{args.gamma:.2f}_" \
               f"lr{args.lr:.0e}_" \
               f"bs{args.batch_size}_" \
               f"uf{args.update_frequency}_" \
               f"buffer{args.buffer_size}_" \
               f"max{args.max_steps_per_episode}_" \
               f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"  
    return filename
