import os
import numpy as np
from common import (load_and_preprocess_data, split_data_with_val,
                    normal_max_absolute_error,build_single_output_dnn,save_best_model,Time)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def stdv_train_dnn(train_times = 20):

    mean_stdv_r2 ,mean_stdv_nmae = 0,0
    print("训练晶粒尺寸标准差模型...")

    best_stdv_r2 = -np.inf  # 初始设为负无穷
    best_stdv_model = None
    best_stdv_pred = None
    best_fact_stdv = None
    max_time = -np.inf
    min_time = np.inf
    
    # 多次训练循环
    for i in range(train_times):
        print(f"\n=== 第 {i+1}/{train_times} 次训练 ===")
        t = Time()
        t.start()

        # 1. 加载和预处理数据
        X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
        
        # 2. 划分数据集并标准化
        X_train_scaled, X_val_scaled, X_test_scaled,y_stdv_train_scaled, y_stdv_val_scaled, y_stdv_test_scaled,\
        y_load_train_scaled, y_load_val_scaled, y_load_test_scaled, scalers = split_data_with_val(X, y_stdv, y_load,0.2,0.25,i + 1)

        stdv_model = build_single_output_dnn(X=X_train_scaled.shape[1])

        callbacks_stdv = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights = True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,min_lr=1e-6)
        ]

        # 训练模型
        history_stdv = stdv_model.fit(
            X_train_scaled, y_stdv_train_scaled,
            validation_data=(X_val_scaled, y_stdv_val_scaled),
            epochs=1000,
            batch_size=16,
            callbacks=callbacks_stdv,
            verbose=0 # 1显示训练过程
        )
        
        # 预测并反标准化
        stdv_pred_scaled = stdv_model.predict(X_test_scaled)
        stdv_pred = scalers['scaler_y_stdv'].inverse_transform(stdv_pred_scaled)
        fact_stdv = scalers['scaler_y_stdv'].inverse_transform(y_stdv_test_scaled.reshape(-1, 1))
        
        # 计算指标
        stdv_r2 = r2_score(fact_stdv, stdv_pred)
        stdv_nmae = normal_max_absolute_error(fact_stdv, stdv_pred)
        mean_stdv_r2 += stdv_r2
        mean_stdv_nmae += stdv_nmae
        
        # 更新最佳模型
        if stdv_r2 > best_stdv_r2:
            best_stdv_r2 = stdv_r2
            best_stdv_model = stdv_model  # 保存模型引用（注意：需确保是深拷贝或重新加载）
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
    save_best_model("stdv",best_stdv_model,best_stdv_r2,best_fact_stdv,best_stdv_pred,scalers,"DNN")
    return (mean_stdv_r2 ,mean_stdv_nmae,max_time,min_time)


def load_train_dnn(train_times = 20):

    mean_load_r2,mean_load_nmae = 0,0
    print("\n训练模具载荷模型...")

    best_load_r2 = -np.inf  # R²越接近1越好，初始设为负无穷
    best_load_model = None
    best_load_pred = None
    best_fact_load = None

    max_time = -np.inf
    min_time = np.inf

    # 多次训练循环
    for i in range(train_times):
        print(f"\n=== 第 {i+1}/{train_times} 次训练 ===")
        t = Time()
        t.start()

        # 1. 加载和预处理数据
        X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
        
        # 2. 划分数据集并标准化
        X_train_scaled, X_val_scaled, X_test_scaled,y_stdv_train_scaled, y_stdv_val_scaled, y_stdv_test_scaled,\
        y_load_train_scaled, y_load_val_scaled, y_load_test_scaled, scalers = split_data_with_val(X, y_stdv, y_load,0.2,0.25,i + 1)
            
        load_model = build_single_output_dnn(input_dim=X_train_scaled.shape[1])
        # 训练模型
        callbacks_load = [
            EarlyStopping(monitor='val_loss', patience=40, restore_best_weights = True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15)
        ]
        history_load = load_model.fit(
            X_train_scaled, y_load_train_scaled,
            validation_data=(X_val_scaled, y_load_val_scaled),
            epochs=1000,
            batch_size=16,
            callbacks=callbacks_load,
            verbose=0 # 1显示训练过程
        )
        
        # 模具载荷模型
        load_pred_scaled = load_model.predict(X_test_scaled)
        load_pred = scalers['scaler_y_load'].inverse_transform(load_pred_scaled)
        fact_load = scalers['scaler_y_load'].inverse_transform(y_load_test_scaled.reshape(-1, 1))
        # 计算指标
        load_r2 = r2_score(fact_load, load_pred)
        load_nmae = normal_max_absolute_error(fact_load, load_pred)
        mean_load_r2 += load_r2
        mean_load_nmae += load_nmae
        
        # 更新最佳模型
        if load_r2 > best_load_r2:
            best_load_r2 = load_r2
            best_load_model = load_model  # 保存模型引用（注意：需确保是深拷贝或重新加载）
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
    save_best_model("load",best_load_model,best_load_r2,best_fact_load,best_load_pred,scalers,"DNN",)
    return  (mean_load_r2,mean_load_nmae,max_time,min_time)
    
def RunDNN(train_times):

    t1 = Time()
    t1.start()
    mean_stdv_r2 ,mean_stdv_nmae,max_stdv_time,min_stdv_time = stdv_train_dnn(train_times)
    t1.stop()
    sum_stdv_time = t1.get_duration("ms")

    t2 = Time()
    t2.start()
    mean_load_r2,mean_load_nmae,max_load_time,min_load_time = load_train_dnn(train_times)
    t2.stop()
    sum_load_time = t2.get_duration("ms")

    return (mean_stdv_r2,mean_stdv_nmae,max_stdv_time,min_stdv_time,sum_stdv_time,
            mean_load_r2,mean_load_nmae,max_load_time,min_load_time,sum_load_time)





