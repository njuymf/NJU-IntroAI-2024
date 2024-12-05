import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler  # 改为随机过采样
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 设置中文字体（例如 SimHei 字体）
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


# 数据预处理
def preprocess_data(data):
    """
    数据预处理，包括：
    - 提取特征和标签；
    - 数据检查、清洗；
    - 特征标准化；
    - 通过随机重复平衡数据（代替 SMOTE）。
    """
    X = []
    y = []

    # 提取特征和标签
    from extract_features import extract_features
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    # 转换为 DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['action'] = y

    print("\n数据集基本信息:")
    print(df.info())
    print(df.describe())

    # 缺失值清理
    if df.isnull().sum().any():
        df = df.dropna()
        print("发现并移除了缺失值")

    # 类别分布检查
    print("原始数据类别分布:")
    print(df['action'].value_counts(normalize=True))

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_names])

    # 随机过采样平衡数据
    print("\n应用随机过采样平衡数据...")
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_scaled, df['action'])

    return X_resampled, y_resampled, feature_names


# 绘制评估图像
def plot_evaluation(y_test, y_pred, y_pred_proba=None):
    """绘制混淆矩阵和 ROC 曲线"""
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    plt.show()

    # ROC 曲线（仅在二分类时绘制）
    if y_pred_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC 曲线 (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真实率')
        plt.title('ROC 曲线')
        plt.legend()
        plt.show()


# 主函数
def main():
    # 加载数据
    from Load_game_records import load_game_records
    data = load_game_records()
    if not data:
        print("未找到可用的游戏记录")
        return

    # 数据预处理
    X_resampled, y_resampled, feature_names = preprocess_data(data)
    if X_resampled is None or y_resampled is None:
        print("数据预处理失败")
        return

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 模型和参数网格定义
    # 模型和参数网格定义
    param_grid = {
        'n_neighbors': [5],  # 仅保留一个邻居数选择
        'weights': ['uniform'],  # 只保留一个权重选项
        'metric': ['euclidean'],  # 只选择一种距离
        'p': [2]  # 只保留欧氏距离
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"\n最佳参数: {grid_search.best_params_}")
    best_knn = grid_search.best_estimator_

    # 模型评估
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 评估图像
    y_pred_proba = best_knn.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
    plot_evaluation(y_test, y_pred, y_pred_proba)

    # 保存模型
    from play import AliensEnvPygame
    env = AliensEnvPygame(level=1, render=False)
    model_path = f'{env.log_folder}/gameplay_model_knn.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_knn, f)

    print(f"\nKNN模型已保存至: {model_path}")


if __name__ == '__main__':
    main()
