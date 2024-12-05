import os
import pickle
import numpy as np
import glob
from sklearn.naive_bayes import GaussianNB  # 导入朴素贝叶斯分类器

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

    clf = GaussianNB()  # 使用朴素贝叶斯分类器
    clf.fit(X, y)

    env = AliensEnvPygame(level=3, render=False)

    # 打印混淆矩阵
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict(X)
    print(confusion_matrix(y, y_pred))

    # 保存模型
    with open(f'{env.log_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")


if __name__ == '__main__':
    main()
