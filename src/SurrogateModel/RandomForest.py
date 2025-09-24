import pandas as pd
import numpy as np
import os
from common import (load_and_preprocess_data,split_data_without_val,
                    normal_max_absolute_error,save_best_model,Time)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录
os.chdir(script_dir)

def stdv_train_RF(train_times = 20):
    mean_stdv_r2 ,mean_stdv_nmae = 0,0
    print("训练晶粒尺寸标准差模型...")
    os.makedirs("../../data/models/RF", exist_ok=True)   
    best_stdv_r2 = -np.inf  # 初始设为负无穷
    best_stdv_model = None
    best_stdv_pred = None
    best_fact_stdv = None

    for i in range(train_times):
        print(f"\n=== 第 {i+1}/{train_times} 次训练 ===")
         # 1. 加载数据
        X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
    
        # 2. 划分数据集并标准化
        X_train_scaled, X_test_scaled, y_stdv_train_scaled, y_stdv_test_scaled, \
        y_load_train_scaled, y_load_test_scaled, scalers = split_data_without_val(X, y_stdv, y_load,0.2,i+1)            

        # 4. 晶粒尺寸标准差模型
        stdv_model = RandomForestRegressor(n_estimators=300, random_state=42)
        stdv_model.fit(X_train_scaled, y_stdv_train_scaled)

        stdv_pred_scaled = stdv_model.predict(X_test_scaled)
        stdv_pred = scalers['scaler_y_stdv'].inverse_transform(stdv_pred_scaled.reshape(-1, 1))
        fact_stdv = scalers['scaler_y_stdv'].inverse_transform(y_stdv_test_scaled.reshape(-1, 1))
        
        # 计算相关指标
        stdv_r2 = r2_score(fact_stdv, stdv_pred)
        stdv_nmae = normal_max_absolute_error(fact_stdv, stdv_pred)
        mean_stdv_r2 += stdv_r2
        mean_stdv_nmae += stdv_nmae

        # 更新最佳模型
        if stdv_r2 > best_stdv_r2:
            best_stdv_r2 = stdv_r2
            best_stdv_model = stdv_model 
            best_stdv_pred = stdv_pred
            best_fact_stdv = fact_stdv     
        # 打印当前结果
        print(f"R²分数: {stdv_r2:.4f}")
        print(f"NMAE: {stdv_nmae:.4f}")
        print("实际值 vs. 预测值(前5行):")
        for j in range(5):
            print(f"实际值: {fact_stdv[j][0]:.4f}, 预测值: {stdv_pred[j][0]:.4f}")

    # 训练结束后，保存最佳模型
    save_best_model("stdv",best_stdv_model,best_stdv_r2,best_fact_stdv,best_stdv_pred,scalers,"RF")
    return (mean_stdv_r2 ,mean_stdv_nmae)   

def load_train_RF(train_times = 20): 

    mean_load_r2 ,mean_load_nmae = 0,0
    print("训练模具载荷模型...")
    os.makedirs("../../data/models/RF", exist_ok=True)   
    best_load_r2 = -np.inf  # 初始设为负无穷
    best_load_model = None
    best_load_pred = None
    best_fact_load = None

    for i in range(train_times):
        print(f"\n=== 第 {i+1}/{train_times} 次训练 ===")
         # 1. 加载数据
        X, y_stdv,y_load = load_and_preprocess_data('../../data/RES-108.txt')
    
        # 2. 划分数据集并标准化
        X_train_scaled, X_test_scaled, y_stdv_train_scaled, y_stdv_test_scaled, \
        y_load_train_scaled, y_load_test_scaled, scalers = split_data_without_val(X, y_stdv, y_load,0.2,i + 1)        

        # 模具最大载荷模型
        load_model = RandomForestRegressor(n_estimators=100, random_state=42)
        load_model.fit(X_train_scaled, y_load_train_scaled)

        load_pred_scaled = load_model.predict(X_test_scaled)
        load_pred = scalers['scaler_y_load'].inverse_transform(load_pred_scaled.reshape(-1, 1))
        fact_load = scalers['scaler_y_load'].inverse_transform(y_load_test_scaled.reshape(-1, 1))
        
        # 计算相关指标
        load_r2 = r2_score(fact_load, load_pred)
        load_nmae = normal_max_absolute_error(fact_load, load_pred)
        mean_load_r2 += load_r2
        mean_load_nmae += load_nmae

        # 更新最佳模型
        if load_r2 > best_load_r2:
            best_load_r2 = load_r2
            best_load_model = load_model 
            best_load_pred = load_pred
            best_fact_load = fact_load     
        # 打印当前结果
        print(f"R²分数: {load_r2:.4f}")
        print(f"NMAE: {load_nmae:.4f}")
        print("实际值 vs. 预测值(前5行):")
        for j in range(5):
            print(f"实际值: {fact_load[j][0]:.4f}, 预测值: {load_pred[j][0]:.4f}")

    # 训练结束后，保存最佳模型
    save_best_model("load",best_load_model,best_load_r2,best_fact_load,best_load_pred,scalers,"RF")
    return (mean_load_r2 ,mean_load_nmae)   

if __name__ == "__main__":
    train_times = 20
    with Time("Kriging 训练STDV时长统计:"):
        mean_stdv_r2 ,mean_stdv_nmae = stdv_train_RF(train_times)
    with Time("Kriging 训练LOAD时长统计:"):
        mean_load_r2,mean_load_nmae = load_train_RF(train_times)

    print(f"晶粒尺寸INFO: r2均值: {mean_stdv_r2 / train_times:.4f}, nmae均值: {mean_stdv_nmae / train_times:.4f}")
    print(f"模具载荷INFO: r2均值: {mean_load_r2 / train_times:.4f}, nmae均值: {mean_load_nmae / train_times:.4f}")
