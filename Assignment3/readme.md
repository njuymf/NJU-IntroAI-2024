# 人工智能导论2024-作业3: Aliens游戏

在当前目录运行 `pip install -r requirements.txt` 安装所需代码库。

安装成功后，可运行 `python play.py` 用键盘方向键玩游戏并存储数据到 `logs/`。可通过修改 `level` 变量为 0~4 设置不同关卡。

存储数据后，修改 `learn.py` 中 `data_list` 变量为存储数据对应的路径，运行 `learn.py` 训练监督学习模型。运行结果将存储在 `logs` 目录下。

修改 `extract_features` 函数完成作业。

# 使用方法


将需要的学习方法从learn文件夹放入主目录下，然后运行`python learn_LSTM.py`即可。

展示结果时，将需要的展示方法从tests文件夹放入主目录下，然后运行`python test_LSTM.py`即可。
