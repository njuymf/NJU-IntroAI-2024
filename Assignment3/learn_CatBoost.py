import os
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix

from extract_features import extract_features
from play import AliensEnvPygame
from Load_game_records import load_game_records
from Load_game_records import inspect_data


def main():
    # 加载游戏记录
    data = load_game_records()
    inspect_data(data)

    if not data:
        print("未找到可用的游戏记录")
        return

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    # 初始化 CatBoostClassifier
    clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=True)
    print("开始训练模型...")

    # 训练模型
    clf.fit(X, y)

    print("模型训练完成")

    # 验证模型 - 打印混淆矩阵
    y_pred = clf.predict(X)
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    # 初始化游戏环境
    env = AliensEnvPygame(level=3, render=False)

    # 保存模型
    model_path = f'{env.log_folder}/gameplay_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f"模型已保存到 {model_path}")


if __name__ == '__main__':
    main()