import pandas as pd
import matplotlib.pyplot as plt


def visualize(csv_path):
    # 读取CSV  
    df = pd.read_csv(csv_path)

    # 绘制训练损失  
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['Episode'], df['Training_Loss'])
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    # 绘制评估回报  
    plt.subplot(1, 2, 2)
    plt.plot(df['Episode'], df['Evaluation_Return'])
    plt.title('Evaluation Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize("datalog\dqn_eps0.90_decay0.99_gamma0.85_lr6e-05_bs64_uf1_buffer64000_max1000_20241126_105537"
              ".csv")
