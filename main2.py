from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from beta_vae import BetaVAE
from exp import train_beta_vae, detect_anomalies_beta_vae, optimize_repair, detect_interval_anomalies

plt.rcParams['font.sans-serif']=['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

if __name__ == "__main__":
    # 1. 从CSV文件加载数据
    original_data = pd.read_csv('U.csv').values[:, 31].reshape(-1, 1)
    
    # 2. 在40%-60%区间添加异常值
    np.random.seed(42)
    n_samples = len(original_data)
    
    # 计算40%和60%的位置索引
    start_idx = int(0.4 * n_samples)
    end_idx = int(0.6 * n_samples)
    middle_range = end_idx - start_idx
    
    # 计算原始数据均值，用于判断异常值方向
    data_mean = np.mean(original_data)
    print(f"数据均值: {data_mean:.4f}")
    
    # 在40%-60%区间添加异常值，根据原始值大小决定异常方向
    noisy_data = original_data.copy()
    for i in range(middle_range):
        idx = start_idx + i
        original_val = original_data[idx][0]
        
        # 根据值的大小方向添加不同的偏移
        if original_val > data_mean:  # 大于均值的点
            # 添加正向偏移，使其更大
            max_deviation = 1 * abs(original_val)  # 使用较大的偏移比例
            deviation = np.abs(np.random.normal(loc=0.7*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] += deviation  # 确保是加上正值
        else:  # 小于均值的点
            # 添加负向偏移，使其更小
            max_deviation = 1 * abs(original_val)
            deviation = np.abs(np.random.normal(loc=0.7*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] -= deviation  # 确保是减去正值
    
    # 统计异常值情况
    larger_count = np.sum((noisy_data > original_data)[start_idx:end_idx])
    smaller_count = np.sum((noisy_data < original_data)[start_idx:end_idx])
    print(f"创建了 {larger_count} 个增大的异常点和 {smaller_count} 个减小的异常点")

    # 创建用于训练的干净数据集（移除40%-60%区间）
    clean_indices = np.ones(n_samples, dtype=bool)
    clean_indices[start_idx:end_idx] = False
    clean_data = original_data[clean_indices]
    print(f"用于训练的干净数据点数量: {len(clean_data)}")
    
    # 创建区间数据和真实异常标签
    interval_noisy_data = noisy_data
    interval_anomalies_true = np.zeros(n_samples, dtype=bool)
    interval_anomalies_true[start_idx:end_idx] = (noisy_data[start_idx:end_idx] != original_data[start_idx:end_idx]).reshape(-1)
    
    # 7. 使用干净数据训练β-VAE模型
    X_train, X_val = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    train_data = torch.FloatTensor(X_train)
    val_data = torch.FloatTensor(X_val)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    
    # 8. 训练β-VAE模型
    model = BetaVAE(input_dim=X_train.shape[1], latent_dim=2)
    train_beta_vae(model, train_loader, epochs=50)
    
    # 9. 在区间数据上进行检测
    print("\n=== 在区间数据上应用异常检测 ===")
    
    # 10. 使用β-VAE进行异常检测
    anomalies_beta_vae, scores_beta_vae, reconstructed_beta_vae = detect_anomalies_beta_vae(
                     model, interval_noisy_data, threshold_percentile=80)
    
    print(f"检测到的异常点总数: {sum(anomalies_beta_vae)}")
    print(f"真实区间异常点数量: {np.sum(interval_anomalies_true)}")
    
    
    # 计算修复效果
    repaired_data_beta_vae, anomaly_indices_beta_vae, anomaly_values_beta_vae, original_values_beta_vae = optimize_repair(
        model, original_data, interval_noisy_data, anomalies_beta_vae)
    
    # 评估异常检测性能
    true_anomalies = interval_anomalies_true.astype(int)
    pred_anomalies = anomalies_beta_vae.astype(int)
    
    accuracy = accuracy_score(true_anomalies, pred_anomalies)
    precision = precision_score(true_anomalies, pred_anomalies, zero_division=0)
    recall = recall_score(true_anomalies, pred_anomalies, zero_division=0)
    f1 = f1_score(true_anomalies, pred_anomalies, zero_division=0)
    
    print("\n=== 异常检测性能评估 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")#所有预测中正确预测的比例
    print(f"精确率 (Precision): {precision:.4f}")#预测为异常的样本中真正异常的比例
    print(f"召回率 (Recall): {recall:.4f}")#真实异常样本中被正确预测出的比例
    print(f"F1分数: {f1:.4f}")#精确率和召回率的调和平均值
    
    # 混淆矩阵
    '''混淆矩阵: TN, FP, FN, TP
    TN: True Negative (正确预测为正常的样本数)
    FP: False Positive (错误地预测为异常的正常样本数（误报）)
    FN: False Negative (错误地预测为正常的异常样本数（漏报）)
    TP: True Positive (正确预测为异常的样本数)'''
    cm = confusion_matrix(true_anomalies, pred_anomalies)
    print("\n混淆矩阵:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # 计算修复误差
    anomaly_mask = anomalies_beta_vae
    if np.any(anomaly_mask):
        repair_mse_beta_vae = np.mean((repaired_data_beta_vae[anomaly_mask] - original_data[anomaly_mask])**2)
        repair_mape_beta_vae = np.mean(np.abs((repaired_data_beta_vae[anomaly_mask] - original_data[anomaly_mask]) / 
                                      np.maximum(0.001, np.abs(original_data[anomaly_mask])))*100)
        
        print(f"\n修复MSE误差: {repair_mse_beta_vae:.6f}")
        print(f"修复MAPE误差: {repair_mape_beta_vae:.6f} %")
    
    # 可视化结果
    plt.figure(figsize=(15, 8))
    plt.title("β-VAE异常检测与修复效果对比 (40%-60%区间)")
    
    # 绘制所有数据点
    plt.plot(np.arange(len(original_data)), original_data[:, 0], 'g-', alpha=0.5, label='原始数据')
    plt.plot(np.arange(len(interval_noisy_data)), interval_noisy_data[:, 0], 'r-', alpha=0.5, label='含异常数据')
    plt.plot(np.arange(len(repaired_data_beta_vae)), repaired_data_beta_vae[:, 0], 'b-', alpha=0.5, label='修复后数据')
    
    # 绘制区间范围竖线
    plt.axvline(x=start_idx, color='purple', linestyle='--', alpha=0.7, label='40%-60%区间')
    plt.axvline(x=end_idx, color='purple', linestyle='--', alpha=0.7)
    
    # 高亮标记被检测的异常点
    plt.scatter(np.where(anomalies_beta_vae)[0], interval_noisy_data[anomalies_beta_vae, 0], 
                facecolors='none', edgecolors='red', s=100, linewidths=1.5, label='检测到的异常点')
    
    # 高亮标记修复前后的对比
    plt.scatter(np.where(anomalies_beta_vae)[0], repaired_data_beta_vae[anomalies_beta_vae, 0], 
                marker='*', color='blue', s=150, label='修复后的点')
    
    # 添加误差线连接异常点和修复后的点
    for i in np.where(anomalies_beta_vae)[0]:
        plt.plot([i, i], [interval_noisy_data[i, 0], repaired_data_beta_vae[i, 0]], 'k--', alpha=0.5)
    
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    
    # 添加文本注释显示修复效果
    if np.any(anomaly_mask):
        plt.figtext(0.15, 0.02, f"修复MSE: {repair_mse_beta_vae:.6f}, 修复MAPE: {repair_mape_beta_vae:.2f}%", 
                    bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # 展示区间局部详细视图
    plt.figure(figsize=(15, 6))
    plt.title("区间异常修复效果详细视图 (40%-60%区间)")
    
    # 只绘制区间数据
    plt.plot(np.arange(start_idx, end_idx), original_data[start_idx:end_idx, 0], 'g-', label='原始数据')
    plt.plot(np.arange(start_idx, end_idx), interval_noisy_data[start_idx:end_idx, 0], 'r-', alpha=0.7, label='含异常数据')
    plt.plot(np.arange(start_idx, end_idx), repaired_data_beta_vae[start_idx:end_idx, 0], 'b-', label='修复后数据')
    
    # 获取区间内的异常点索引
    anomalies_in_interval = np.where(anomalies_beta_vae[start_idx:end_idx])[0] + start_idx
    
    # 标记区间内的异常点
    if len(anomalies_in_interval) > 0:
        plt.scatter(anomalies_in_interval, interval_noisy_data[anomalies_in_interval, 0], 
                    facecolors='none', edgecolors='red', s=100, linewidths=1.5, label='检测到的异常点')
        plt.scatter(anomalies_in_interval, repaired_data_beta_vae[anomalies_in_interval, 0], 
                    marker='*', color='blue', s=150, label='修复后的点')
    
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()