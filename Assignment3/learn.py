import os
import pickle
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier

from extract_features import extract_features
from extract_features import extract_features_plus
from play import AliensEnvPygame
from Load_game_records import load_game_records
from Load_game_records import inspect_data


def main():
    print("当前工作目录:", os.getcwd())
    # 加载游戏记录
    data = load_game_records()
    # inspect_data(data)

    if not data:
        print("未找到可用的游戏记录")
        return

    X = []
    y = []
    for observation, action in data:
        features = extract_features_plus(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    env = AliensEnvPygame(level=1, render=False)

    # 打印混淆矩阵
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict(X)
    print("混淆矩阵：")
    print(confusion_matrix(y, y_pred))
    
    # 打印准确率
    from sklearn.metrics import accuracy_score
    print("准确率：",end="")
    # 保留两位小数,写成百分数
    print('%.2f%%' % (accuracy_score(y, y_pred) * 100))

    # 保存模型

    with open(f'{env.log_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")


if __name__ == '__main__':
    main()
