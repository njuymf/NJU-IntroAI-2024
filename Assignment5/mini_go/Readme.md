# Mini-AlphaGo 复现 #

## 使用方法

运行环境在requirements.txt中，
在当前目录运行 `pip install -r requirements.txt` 安装所需代码库。

安装成功后，可运行 `python training.py` 进行训练。训练过程大约需要 7 小时，训练过程中会保存模型参数到 `./parameters[0-3]` 目录下,具体参见`python save_model`。

存储数据后，运行 `python AlphaGo_vs_random.py` 进行测试。