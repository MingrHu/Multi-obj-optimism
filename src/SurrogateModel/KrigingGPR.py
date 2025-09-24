import numpy as np
import os
from common import (load_and_preprocess_data,split_data_without_val,
                    normal_max_absolute_error,save_best_model,Time)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录
os.chdir(script_dir)

def stdv_train_kriging(train_times = 20):

    mean_stdv_r2 ,mean_stdv_nmae = 0,0
    print("训练晶粒尺寸标准差模型...")
    os.makedirs("../../data/models/Kriging", exist_ok=True)
    best_stdv_r2 = -np.inf  # 初始设为负无穷
    best_stdv_model = None
    best_stdv_pred = None
    best_fact_stdv = None

    max_time = -np.inf
    min_time = np.inf

    for i in range(train_times):
        print(f"\n=== 第 {i+1}/{train_times} 次训练 ===")
        t = Time()
        t.start()

         # 1. 加载数据
        X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
    
        # 2. 划分数据集并标准化
        X_train_scaled, X_test_scaled, y_stdv_train_scaled, y_stdv_test_scaled, \
        y_load_train_scaled, y_load_test_scaled, scalers = split_data_without_val(X, y_stdv, y_load,0.2,i+1)
        # 3. 定义 Kriging 模型
        kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4))
        # 晶粒尺寸标准差模型
        stdv_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=20,
            alpha=0.1,
            random_state=42
        )
        # 训练模型
        stdv_model.fit(X_train_scaled, y_stdv_train_scaled)
        
        stdv_pred_scaled = stdv_model.predict(X_test_scaled)
        stdv_pred = scalers['scaler_y_stdv'].inverse_transform(stdv_pred_scaled.reshape(-1, 1))
        fact_stdv = scalers['scaler_y_stdv'].inverse_transform(y_stdv_test_scaled.reshape(-1, 1))

        # 计算相关指标
        stdv_r2 = r2_score(y_stdv_test_scaled, stdv_pred_scaled) 
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
        t.stop()
        max_time = max(max_time,t.get_duration("ms"))
        min_time = min(min_time,t.get_duration("ms"))

    # 训练结束后，保存最佳模型
    save_best_model("stdv",best_stdv_model,best_stdv_r2,best_fact_stdv,best_stdv_pred,scalers,"Kriging")
    return (mean_stdv_r2 ,mean_stdv_nmae,max_time,min_time)  


def load_train_kriging(train_times = 20):
    mean_load_r2,mean_load_nmae = 0, 0
    print("训练模具最大载荷模型...")
    os.makedirs("../../data/models/Kriging", exist_ok=True)
    best_load_r2 = -np.inf  # 初始设为负无穷
    best_load_model = None
    best_load_pred = None
    best_fact_load = None

    max_time = -np.inf
    min_time = np.inf

    for i in range(train_times):
        print(f"\n=== 第 {i+1}/{train_times} 次训练 ===")
        t = Time()
        t.start()

         # 1. 加载数据
        X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
    
        # 2. 划分数据集并标准化
        X_train_scaled, X_test_scaled, y_stdv_train_scaled, y_stdv_test_scaled, \
        y_load_train_scaled, y_load_test_scaled, scalers = split_data_without_val(X, y_stdv, y_load,0.2,i + 1)
        # 3. 定义 Kriging 模型
        kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4))
        # 模具载荷模型
        load_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=20,
            alpha=0.1,
            random_state=42
        )
        # 训练模型
        load_model.fit(X_train_scaled, y_load_train_scaled)

        load_pred_scaled = load_model.predict(X_test_scaled)
        load_pred = scalers['scaler_y_load'].inverse_transform(load_pred_scaled.reshape(-1, 1))
        fact_load = scalers['scaler_y_load'].inverse_transform(y_load_test_scaled.reshape(-1, 1))
        
        load_r2 = r2_score(y_load_test_scaled, load_pred_scaled)  
        load_nmae = normal_max_absolute_error(fact_load,load_pred)
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
        t.stop()
        max_time = max(max_time,t.get_duration("ms"))
        min_time = min(min_time,t.get_duration("ms"))

    # 训练结束后，保存最佳模型
    save_best_model("load",best_load_model,best_load_r2,best_fact_load,best_load_pred,scalers,"Kriging")
    return  (mean_load_r2,mean_load_nmae,max_time,min_time)  

def RunKriging(train_times):

    t1 = Time()
    t1.start()
    mean_stdv_r2 ,mean_stdv_nmae,max_stdv_time,min_stdv_time = stdv_train_kriging(train_times)
    t1.stop()
    sum_stdv_time = t1.get_duration("ms")

    t2 = Time()
    t2.start()
    mean_load_r2,mean_load_nmae,max_load_time,min_load_time = load_train_kriging(train_times)
    t2.stop()
    sum_load_time = t2.get_duration("ms")

    return (mean_stdv_r2,mean_stdv_nmae,max_stdv_time,min_stdv_time,sum_stdv_time,
            mean_load_r2,mean_load_nmae,max_load_time,min_load_time,sum_load_time)