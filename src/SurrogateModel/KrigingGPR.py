import pandas as pd
import numpy as np
import joblib
import os
from common import load_and_preprocess_data,split_data_without_val
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录
os.chdir(script_dir)

def Kriging():
    # 1. 加载数据
    X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
    
    # 2. 划分数据集并标准化
    X_train_scaled, X_test_scaled, y_stdv_train_scaled, y_stdv_test_scaled, \
    y_load_train_scaled, y_load_test_scaled, scalers = split_data_without_val(X, y_stdv, y_load)
    
    # 3. 定义 Kriging 模型
    kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4))
    
    # 晶粒尺寸标准差模型
    stdv_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        alpha=0.1,
        random_state=42
    )
    stdv_model.fit(X_train_scaled, y_stdv_train_scaled)
    stdv_pred_scaled, stdv_std = stdv_model.predict(X_test_scaled, return_std=True)
    stdv_pred = scalers['scaler_y_stdv'].inverse_transform(stdv_pred_scaled.reshape(-1, 1)).ravel()
    stdv_r2 = r2_score(y_stdv_test_scaled, stdv_pred_scaled)  # 注意：这里应比较标准化后的值
    
    # 模具载荷模型
    load_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        alpha=0.1,
        random_state=42
    )
    load_model.fit(X_train_scaled, y_load_train_scaled)
    load_pred_scaled, load_std = load_model.predict(X_test_scaled, return_std=True)
    load_pred = scalers['scaler_y_load'].inverse_transform(load_pred_scaled.reshape(-1, 1)).ravel()
    load_r2 = r2_score(y_load_test_scaled, load_pred_scaled)  # 注意：这里应比较标准化后的值
    
    # 4. 保存模型
    os.makedirs("../../data/models/Kriging", exist_ok=True)
    joblib.dump(stdv_model, '../../data/models/Kriging/stdv_model.pkl')
    joblib.dump(load_model, '../../data/models/Kriging/die_load_model.pkl')
    joblib.dump(scalers, '../../data/models/Kriging/scalers.pkl')
    
    # 5. 输出结果
    print("=== 晶粒尺寸标准差 ===")
    print(f"R²分数: {stdv_r2:.4f}")
    print("\n实际值 vs. 预测值(前5行):")
    print(pd.DataFrame({
        '实际值': scalers['scaler_y_stdv'].inverse_transform(y_stdv_test_scaled[:5].reshape(-1, 1)).ravel(),
        '预测值': stdv_pred[:5]
    }))
    
    print("\n=== 模具载荷模型 ===")
    print(f"R²分数: {load_r2:.4f}")
    print("\n实际值 vs. 预测值(前5行):")
    print(pd.DataFrame({
        '实际值': scalers['scaler_y_load'].inverse_transform(y_load_test_scaled[:5].reshape(-1, 1)).ravel(),
        '预测值': load_pred[:5]
    }))


if __name__ == "__main__":
    Kriging()