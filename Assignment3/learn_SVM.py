import os
import pickle
import numpy as np
from sklearn.svm import SVC
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

        # 提取特征和标签
    X = []
    y = []
    print("开始特征提取...")
    for idx, (observation, action) in enumerate(data):
        features = extract_features(observation)
        X.append(features)
        y.append(action)

        if (idx + 1) % 100 == 0:  # 每100个记录输出一次
            print(f"提取特征: 已处理 {idx + 1} 条记录")

    X = np.array(X)
    y = np.array(y)
    print(f"特征提取完成: 共提取 {X.shape[0]} 条记录，特征维度为 {X.shape[1]}")

    # 使用支持向量机进行训练
    clf = SVC(kernel='rbf')  # 可以更改为其他核函数
    print("开始训练支持向量机模型...")
    clf.fit(X, y)
    print("模型训练完成")

    env = AliensEnvPygame(level=3, render=False)

    # 打印预测结果
    y_pred = clf.predict(X)
    print("模型预测完成")

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(y, y_pred)
    print("混淆矩阵:")
    print(conf_matrix)

    # 输出预测结果统计
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    print("预测类别分布:")
    for cls, count in zip(unique_classes, counts):
        print(f"类别 {cls}: {count} 次")

        # 保存模型
    with open(f'{env.log_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型保存完成。")


if __name__ == '__main__':
    main()