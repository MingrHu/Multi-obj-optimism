import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录
os.chdir(script_dir)

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据加载与预处理函数
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
                  'STDV_grainSize', 'Die_Load']
    
    # 特征与目标变量
    X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']].values
    y_stdv = df['STDV_grainSize'].values.reshape(-1, 1)
    y_load = df['Die_Load'].values.reshape(-1, 1)
    
    return X, y_stdv, y_load

# 2. 数据标准化和划分函数
def prepare_data(X, y_stdv, y_load, test_size=0.2, val_size=0.5, random_state=42):
    """数据标准化和划分"""
    # 特征标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 目标变量标准化
    scaler_y_stdv = StandardScaler()
    y_stdv_scaled = scaler_y_stdv.fit_transform(y_stdv)
    
    scaler_y_load = StandardScaler()
    y_load_scaled = scaler_y_load.fit_transform(y_load)
    
    # 数据划分
    X_train, X_temp, y_train_stdv, y_temp_stdv, y_train_load, y_temp_load = train_test_split(
        X_scaled, y_stdv_scaled, y_load_scaled, test_size=test_size, random_state=random_state
    )
    
    X_val, X_test, y_val_stdv, y_test_stdv, y_val_load, y_test_load = train_test_split(
        X_temp, y_temp_stdv, y_temp_load, test_size=val_size, random_state=random_state
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    y_train_stdv_tensor = torch.FloatTensor(y_train_stdv).to(device)
    y_val_stdv_tensor = torch.FloatTensor(y_val_stdv).to(device)
    y_test_stdv_tensor = torch.FloatTensor(y_test_stdv).to(device)
    
    y_train_load_tensor = torch.FloatTensor(y_train_load).to(device)
    y_val_load_tensor = torch.FloatTensor(y_val_load).to(device)
    y_test_load_tensor = torch.FloatTensor(y_test_load).to(device)
    
    scalers = {
        'X': scaler_X,
        'y_stdv': scaler_y_stdv,
        'y_load': scaler_y_load
    }
    
    return (X_train_tensor, X_val_tensor, X_test_tensor,
            y_train_stdv_tensor, y_val_stdv_tensor, y_test_stdv_tensor,
            y_train_load_tensor, y_val_load_tensor, y_test_load_tensor,
            scalers)

# 3. 定义DNN模型
class MultiOutputDNN(nn.Module):
    """多输出深度神经网络"""
    # 隐藏层3层 64 32 16
    # 丢弃率 0.2 0.1
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rates=[0.2, 0.1]):
        super(MultiOutputDNN, self).__init__()
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            # 输入64维
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rates[0]),
            # 输入32维
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout_rates[1]),
        )
        
        # 晶粒尺寸标准差输出分支
        self.grain_size_branch = nn.Sequential(
            # 32维到16维
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            # 16维到1维
            nn.Linear(hidden_dims[2], 1)
        )
        
        # 模具载荷输出分支
        self.mold_load_branch = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
    
    # 前向传播
    def forward(self, x):
        shared_features = self.shared_layers(x)
        grain_size = self.grain_size_branch(shared_features)
        mold_load = self.mold_load_branch(shared_features)
        return grain_size, mold_load

# 4. 训练器定义
class MultiOutputTrainer:
    """多输出模型训练器"""
    def __init__(self, model, lr=0.001, weight_decay=1e-5):
        self.model = model.to(device)   # 选择GPU/CPU
        self.criterion = nn.MSELoss()   # 均方误差 回归任务
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)   # 优化器 默认都是Adam
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience= 5,
        ) # 学习率的设定
        self.train_losses = [] # 记录训练损失
        self.val_losses = []    # 验证集损失
    
    def train_epoch(self, train_loader, loss_weights=[0.5, 0.5]):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_stdv_batch, y_load_batch in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            pred_stdv, pred_load = self.model(X_batch)
            
            # 计算损失
            loss_stdv = self.criterion(pred_stdv, y_stdv_batch)
            loss_load = self.criterion(pred_load, y_load_batch)
            loss = loss_weights[0] * loss_stdv + loss_weights[1] * loss_load
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, loss_weights=[0.5, 0.5]):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_stdv_batch, y_load_batch in val_loader:
                pred_stdv, pred_load = self.model(X_batch)
                
                loss_stdv = self.criterion(pred_stdv, y_stdv_batch)
                loss_load = self.criterion(pred_load, y_load_batch)
                loss = loss_weights[0] * loss_stdv + loss_weights[1] * loss_load
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=200, patience=50, loss_weights=[0.2, 0.8]):
        """完整训练过程"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, loss_weights)
            val_loss = self.validate(val_loader, loss_weights)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if epoch % 1 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        os.remove('best_model.pth')
        print('Training completed!')
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            pred_stdv, pred_load = self.model(X)
        return pred_stdv.cpu().numpy(), pred_load.cpu().numpy()

# 5. 主函数
def main():
    # 加载数据
    X, y_stdv, y_load = load_and_preprocess_data('../../data/RES.txt')
    
    # 准备数据
    (X_train, X_val, X_test,
     y_train_stdv, y_val_stdv, y_test_stdv,
     y_train_load, y_val_load, y_test_load,
     scalers) = prepare_data(X, y_stdv, y_load)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train_stdv, y_train_load)
    val_dataset = TensorDataset(X_val, y_val_stdv, y_val_load)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    input_dim = X_train.shape[1]
    model = MultiOutputDNN(input_dim)
    
    # 创建训练器并训练
    trainer = MultiOutputTrainer(model, lr=0.001)
    trainer.train(train_loader, val_loader, epochs=200, patience=15)
    
    # 预测
    stdv_pred_scaled, load_pred_scaled = trainer.predict(X_test)
    
    # 反标准化
    stdv_pred = scalers['y_stdv'].inverse_transform(stdv_pred_scaled)
    load_pred = scalers['y_load'].inverse_transform(load_pred_scaled)
    
    fact_stdv = scalers['y_stdv'].inverse_transform(y_test_stdv.cpu().numpy())
    fact_load = scalers['y_load'].inverse_transform(y_test_load.cpu().numpy())
    
    # 计算R²分数
    stdv_r2 = r2_score(fact_stdv, stdv_pred)
    load_r2 = r2_score(fact_load, load_pred)
    
    # 保存模型和标准化器
    os.makedirs("../models/DNN_Optimized_PyTorch", exist_ok=True)
    torch.save(model.state_dict(), '../models/DNN_Optimized_PyTorch/multi_output_model.pth')
    joblib.dump(scalers['X'], '../models/DNN_Optimized_PyTorch/scaler_X.pkl')
    joblib.dump(scalers['y_stdv'], '../models/DNN_Optimized_PyTorch/scaler_y_stdv.pkl')
    joblib.dump(scalers['y_load'], '../models/DNN_Optimized_PyTorch/scaler_y_load.pkl')
    
    # 输出结果
    print("\n=== 晶粒尺寸标准差模型 ===")
    print(f"R²分数: {stdv_r2:.4f}")
    print("\n实际值 vs. 预测值 前5行:")
    for i in range(5):
        print(f"实际值: {fact_stdv[i][0]:.4f}, 预测值: {stdv_pred[i][0]:.4f}")

    print("\n=== 模具载荷模型 ===")
    print(f"R²分数: {load_r2:.4f}")
    print("\n实际值 vs. 预测值 前5行:")
    for i in range(5):
        print(f"实际值: {fact_load[i][0]:.4f}, 预测值: {load_pred[i][0]:.4f}")

if __name__ == "__main__":
    main()