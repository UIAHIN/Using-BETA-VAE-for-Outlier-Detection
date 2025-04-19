from beta_vae import BetaVAE
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_beta_vae(model, train_loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 获取模型的输出
            recon_batch, mu, log_var = model(data)
            
            # 修改这一行，确保传递所有需要的参数
            loss = model.loss_function(data, recon_batch, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_loss = train_loss / len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Average loss: {avg_loss:.6f}')

def detect_anomalies_beta_vae(model, data, threshold_percentile=95):
    """
    使用β-VAE模型检测异常
    
    参数:
        model: 训练好的β-VAE模型
        data: 待检测的数据
        threshold_percentile: 异常阈值百分位数
        
    返回:
        anomalies: 布尔数组，标记异常点
        scores: 重构误差分数
        reconstructed: 重构后的数据
    """
    # 转换为torch张量
    if isinstance(data, np.ndarray):
        data_tensor = torch.FloatTensor(data)
    else:
        data_tensor = data
    
    # 获取重构误差分数
    scores = model.get_reconstruction_scores(data)
    
    # 计算阈值
    threshold = np.percentile(scores, threshold_percentile)
    
    # 标记异常点
    anomalies = scores > threshold
    
    # 获取重构数据
    model.eval()
    with torch.no_grad():
        reconstructed, _, _ = model(data_tensor)
        reconstructed = reconstructed.numpy() if isinstance(data, np.ndarray) else reconstructed
    
    return anomalies, scores, reconstructed

def detect_interval_anomalies(model, data, start_idx, end_idx, threshold_percentile=95):
    """
    在数据的特定区间内检测异常值
    
    参数:
        model: 训练好的模型
        data: 待检测的数据
        start_idx: 区间开始索引
        end_idx: 区间结束索引
        threshold_percentile: 异常阈值百分位数
        
    返回:
        全局异常检测结果（只在区间内标记），异常分数，重构数据
    """
    # 转换为torch张量
    if isinstance(data, np.ndarray):
        data_tensor = torch.FloatTensor(data)
    else:
        data_tensor = data
    
    # 获取重构误差分数
    scores = model.get_reconstruction_scores(data)
    
    # 只考虑区间内的分数来计算阈值
    interval_scores = scores[start_idx:end_idx]
    threshold = np.percentile(interval_scores, threshold_percentile)
    
    # 创建全局异常标记（初始全为False）
    anomalies = np.zeros(len(data), dtype=bool)
    
    # 只在指定区间内标记异常
    anomalies[start_idx:end_idx] = scores[start_idx:end_idx] > threshold
    
    # 获取重构数据
    model.eval()
    with torch.no_grad():
        reconstructed, _, _ = model(data_tensor)
        reconstructed = reconstructed.numpy() if isinstance(data, np.ndarray) else reconstructed
    
    return anomalies, scores, reconstructed

def optimize_repair(model, original_data, noisy_data, anomalies):
    """
    使用β-VAE模型修复异常数据点
    
    参数:
        model: 训练好的β-VAE模型
        original_data: 原始数据（用于比较）
        noisy_data: 含有异常的数据
        anomalies: 布尔数组，标记异常点
        
    返回:
        repaired_data: 修复后的数据
        anomaly_indices: 异常点的索引
        anomaly_values: 异常点的原始值
        original_values: 异常点对应的真实值
    """
    repaired_data = noisy_data.copy()
    
    # 获取异常点的索引
    anomaly_indices = np.where(anomalies)[0]
    
    # 保存异常值和原始值
    anomaly_values = noisy_data[anomalies].copy()
    original_values = original_data[anomalies].copy()
    
    # 将异常数据转换为张量
    data_tensor = torch.FloatTensor(noisy_data)
    
    # 使用模型重建数据
    model.eval()
    with torch.no_grad():
        reconstructed, _, _ = model(data_tensor)
        reconstructed = reconstructed.numpy()
    
    # 只替换异常点的值
    repaired_data[anomalies] = reconstructed[anomalies]
    
    return repaired_data, anomaly_indices, anomaly_values, original_values