from torch import nn
import torch
import numpy as np

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=4):
        super(BetaVAE, self).__init__()
        
        # 保存潜在空间维度为实例属性
        self.latent_dim = latent_dim
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim * 2)  # 输出 mu 和 log_var
        )
        
        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )
        
        self.beta = beta  # β值控制KL散度的权重

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var

    def loss_function(self, x, recon_x, mu, log_var):
        # 使用MSE损失代替二元交叉熵
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL散度部分保持不变
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return MSE + self.beta * KLD

    def get_reconstruction_scores(self, data):
        """
        计算输入数据的重构误差分数
        
        参数:
            data: 输入数据，可能是numpy数组或者张量
            
        返回:
            每个数据点的重构误差分数(越高表示越可能是异常)
        """
        # 确保输入是张量
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
            
        # 将模型设置为评估模式
        self.eval()
        
        with torch.no_grad():
            # 前向传播获取重构结果
            recon_batch, _, _ = self(data)
            
            # 计算每个数据点的均方误差作为重构分数
            scores = torch.mean((recon_batch - data) ** 2, dim=1).numpy()
            
        return scores

def train_beta_vae(model, train_loader, epochs, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0]
            recon_x, mu, log_var = model(x)
            loss = model.loss_function(x, recon_x, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}')