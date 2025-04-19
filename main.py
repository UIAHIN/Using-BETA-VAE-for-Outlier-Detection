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
from exp import train_beta_vae, detect_anomalies_beta_vae, optimize_repair

plt.rcParams['font.sans-serif']=['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

if __name__ == "__main__":
    # 1. 从CSV文件加载数据
    original_data = pd.read_csv('U.csv').values[:, 31].reshape(-1, 1)
    
    
    # 2. 计算原始数据均值
    data_mean = np.mean(original_data)
    
    # 3. 确定异常值范围（均值±0.5%）
    threshold = 0.005 * abs(data_mean)
    lower_bound = data_mean - threshold
    upper_bound = data_mean + threshold
    
    # 4. 标记原始数据中的异常点
    original_anomalies = (original_data < lower_bound) | (original_data > upper_bound)
    print(f"原始数据中已有 {np.sum(original_anomalies)} 个自然异常点")
    
    # 5. 创建干净数据集（移除异常值）用于训练
    clean_indices = ~original_anomalies.flatten()  # 取非异常点索引
    clean_data = original_data[clean_indices]
    print(f"用于训练的干净数据点数量: {len(clean_data)}")
    
    # 6. 创建带异常值的数据集用于后续检测（使大值更大，小值更小）
    np.random.seed(42)
    noisy_data = original_data.copy()
    anomaly_indices = np.where(original_anomalies.flatten())[0]
    
    for idx in anomaly_indices:
        original_val = original_data[idx][0]
        # 根据值的大小方向添加不同的偏移
        if original_val > data_mean:  # 大于均值的点
            max_deviation = 1 * abs(original_val)  # 使用更大的偏移比例
            deviation = np.abs(np.random.normal(loc=0.4*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] += deviation  # 确保是加上正值
        else:  # 小于均值的点
            max_deviation = 1 * abs(original_val)  # 使用更大的偏移比例
            deviation = np.abs(np.random.normal(loc=0.4*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] -= deviation  # 确保是减去正值
    
    # 输出信息，了解异常值修改的程度
    larger_anomalies = np.sum(noisy_data > original_data)
    smaller_anomalies = np.sum(noisy_data < original_data)
    print(f"创建了 {larger_anomalies} 个增大的异常点和 {smaller_anomalies} 个减小的异常点")
    
    # 7. 使用干净数据训练β-VAE模型
    X_train, X_val = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    train_data = torch.FloatTensor(X_train)
    val_data = torch.FloatTensor(X_val)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    
    # 8. 训练β-VAE模型
    model = BetaVAE(input_dim=X_train.shape[1], latent_dim=2)
    train_beta_vae(model, train_loader, epochs=50)
    
    # 9. 在未知数据（含有异常）上进行检测
    print("\n=== 在未知数据上应用异常检测 ===")
    
    # 10. 使用β-VAE进行异常检测
    anomalies_beta_vae, scores_beta_vae, reconstructed_beta_vae = detect_anomalies_beta_vae(
                     model, noisy_data, threshold_percentile=95.95)
    
    data_mean = np.mean(original_data)
    threshold = 0.005 * np.abs(data_mean)
    original_anomalies = (original_data < (data_mean - threshold)) | (original_data > (data_mean + threshold))
    print(f"实际原始异常点数量: {np.sum(original_anomalies)}")
    
    print("\n=== 检测方法比较 ===")
    print(f"β-VAE检测到的异常点: {sum(anomalies_beta_vae)}")
    
    # 尝试优化阈值
    print("\n=== 尝试优化检测阈值 ===")
    best_f1 = 0
    best_threshold_percentile = 95.95  # 初始值
    
    for percentile in np.linspace(90, 99, 10):
        test_anomalies, _, _ = detect_anomalies_beta_vae(model, noisy_data, threshold_percentile=percentile)
        test_f1 = f1_score(original_anomalies.flatten().astype(int), test_anomalies.astype(int), zero_division=0)
        print(f"阈值百分位: {percentile:.2f}, F1分数: {test_f1:.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_threshold_percentile = percentile
    
    print(f"\n最佳阈值百分位: {best_threshold_percentile:.2f}, F1分数: {best_f1:.4f}")
    
    # 使用最佳阈值重新检测
    anomalies_beta_vae, scores_beta_vae, reconstructed_beta_vae = detect_anomalies_beta_vae(
                     model, noisy_data, threshold_percentile=best_threshold_percentile)
    
    # 计算修复效果
    repaired_data_beta_vae, anomaly_indices_beta_vae, anomaly_values_beta_vae, original_values_beta_vae = optimize_repair(
        model, original_data, noisy_data, anomalies_beta_vae)

    repair_mse_beta_vae = np.mean((repaired_data_beta_vae[anomalies_beta_vae] - original_data[anomalies_beta_vae])**2)
    repair_mape_beta_vae = np.mean(np.abs((repaired_data_beta_vae[anomalies_beta_vae] - original_data[anomalies_beta_vae]) / original_data[anomalies_beta_vae])*100)
    print(f"β-VAE自动检测-修复MSE误差: {repair_mse_beta_vae:.6f}")
    print(f"β-VAE自动检测-修复MAPE误差: {repair_mape_beta_vae:.6f} %")
    
    plt.figure(figsize=(10, 6))
    plt.title("β-VAE自动异常检测结果")
    plt.scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c=scores_beta_vae, cmap='coolwarm', alpha=0.6)
    plt.scatter(np.where(anomalies_beta_vae)[0], noisy_data[anomalies_beta_vae, 0], 
                facecolors='none', edgecolors='black', s=100, linewidths=1.5, label='β-VAE检测异常')
    plt.colorbar(label='异常分数')
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 在现有绘图代码后添加
    # 创建修复效果对比图
    plt.figure(figsize=(15, 8))
    plt.title("β-VAE异常检测与修复效果对比")
    
    # 绘制所有数据点
    plt.plot(np.arange(len(original_data)), original_data[:, 0], 'g-', alpha=0.5, label='原始数据')
    plt.plot(np.arange(len(noisy_data)), noisy_data[:, 0], 'r-', alpha=0.5, label='含异常数据')
    plt.plot(np.arange(len(repaired_data_beta_vae)), repaired_data_beta_vae[:, 0], 'b-', alpha=0.5, label='修复后数据')
    
    # 高亮标记被检测的异常点
    plt.scatter(np.where(anomalies_beta_vae)[0], noisy_data[anomalies_beta_vae, 0], 
                facecolors='none', edgecolors='red', s=100, linewidths=1.5, label='检测到的异常点')
    
    # 高亮标记修复前后的对比
    plt.scatter(np.where(anomalies_beta_vae)[0], repaired_data_beta_vae[anomalies_beta_vae, 0], 
                marker='*', color='blue', s=150, label='修复后的点')
    
    # 添加误差线连接异常点和修复后的点
    for i in np.where(anomalies_beta_vae)[0]:
        plt.plot([i, i], [noisy_data[i, 0], repaired_data_beta_vae[i, 0]], 'k--', alpha=0.5)
    
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    
    # 添加文本注释显示修复效果
    plt.figtext(0.15, 0.02, f"修复MSE: {repair_mse_beta_vae:.6f}, 修复MAPE: {repair_mape_beta_vae:.2f}%", 
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # 展示局部细节图 - 选择前50个点作为示例
    sample_size = min(50, len(original_data))
    plt.figure(figsize=(15, 6))
    plt.title(f"修复效果细节展示 (前{sample_size}个点)")
    
    plt.plot(np.arange(sample_size), original_data[:sample_size, 0], 'g-', label='原始数据')
    plt.plot(np.arange(sample_size), noisy_data[:sample_size, 0], 'r-', alpha=0.7, label='含异常数据')
    plt.plot(np.arange(sample_size), repaired_data_beta_vae[:sample_size, 0], 'b-', label='修复后数据')
    
    # 获取前sample_size个样本中的异常点索引
    anomalies_indices_in_range = np.where(anomalies_beta_vae[:sample_size])[0]
    
    # 标记这个区域内的异常点
    if len(anomalies_indices_in_range) > 0:
        plt.scatter(anomalies_indices_in_range, noisy_data[anomalies_indices_in_range, 0], 
                    facecolors='none', edgecolors='red', s=100, linewidths=1.5)
        plt.scatter(anomalies_indices_in_range, repaired_data_beta_vae[anomalies_indices_in_range, 0], 
                    marker='*', color='blue', s=150)
    
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 评估异常检测性能
    # 将numpy布尔数组转换为int类型供sklearn使用
    true_anomalies = original_anomalies.flatten().astype(int)
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