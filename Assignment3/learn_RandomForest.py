import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from play import AliensEnvPygame
import matplotlib
from Load_game_records import load_game_records

# 设置中文字体（例如使用 SimHei 字体）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def main():
    # 加载游戏记录
    data = load_game_records()

    if not data:
        print("未找到可用的游戏记录")
        return

    # 提取特征和标签
    X = []
    y = []
    for observation, action in data:
        from extract_features import extract_features
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    # 将数据转换为 DataFrame，便于后续处理
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['action'] = y

    # 数据检查
    print("\n数据集基本信息:")
    print(df.info())
    print(df.describe())

    # 检查缺失值
    if df.isnull().sum().any():
        df = df.dropna()
        print("发现并移除了缺失值")

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_names])
    """
    # 处理类别不平衡
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, df['action'])
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # 定义参数网格进行调优
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'max_features': ['log2', 'sqrt'],
        'min_samples_split': [2, 5, 10]
    }

    # 使用网格搜索和交叉验证
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_

    # 交叉验证评分
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    print(f"\n交叉验证平均准确率: {cv_scores.mean():.2%}")

    # 在测试集上评估模型
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    plt.show()

    # 绘制 ROC 曲线
    if len(np.unique(y_test)) == 2:
        y_scores = best_rf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        auc_score = roc_auc_score(y_test, y_scores)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC 曲线 (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真实率')
        plt.title('ROC 曲线')
        plt.legend()
        plt.show()

    # 保存模型
    env = AliensEnvPygame(level=4, render=False)
    model_path = f'{env.log_folder}/gameplay_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf, f)

    print(f"\n模型已保存至: {model_path}")


if __name__ == '__main__':
    main()
