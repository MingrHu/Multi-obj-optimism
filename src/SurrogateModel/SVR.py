import joblib
import numpy as np
import os
from common import (load_and_preprocess_data,split_data_without_val,
                    normal_max_absolute_error,save_best_model,Time)
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
# exp 目前支持任意参数输入 下面的例子是
# 1-7 是自变量 X矩阵 后面的三个是因变量Y矩阵
# vars_out是为了标识不同的列取的名字 同时用于控制输入输出位置
# 
vars_out = ["1","2","3","4","5","6","7","res1","res2","res3"]
file = '/Users/hmr/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'

def svr_fun():
         # 1. 加载数据
        X, Y = load_and_preprocess_data(file,vars_out,7)
    
        # 2. 划分数据集并标准化
        (X_train_scaled, X_test_scaled,
        Y_train_scaled_list, Y_test_scaled_list,
        scalers) = split_data_without_val(X, Y)            

        for idx in range(len(Y_train_scaled_list)):
            cur_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            cur_model.fit(X_train_scaled, Y_train_scaled_list[idx])
            pred_scaled = cur_model.predict(X_test_scaled)
            test_scaled = Y_test_scaled_list[idx]
            # 调用标准化器
            pred = scalers[f'scaler_y_{idx}'].inverse_transform(pred_scaled.reshape(-1, 1))
            fact = scalers[f'scaler_y_{idx}'].inverse_transform(test_scaled.reshape(-1, 1))
        
            # 计算相关指标
            stdv_r2 = r2_score(fact, pred)
            stdv_nmae = normal_max_absolute_error(fact, pred)

            # 打印当前结果
            print(f"R²分数: {stdv_r2:.4f}")
            print(f"NMAE: {stdv_nmae:.4f}")
            print("实际值 vs. 预测值(前5行):")
            for j in range(5):
                print(f"实际值: {fact[j][0]:.4f}, 预测值: {pred[j][0]:.4f}")

if __name__ == "__main__":
     svr_fun()