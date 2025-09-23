import os
import tensorflow as tf
from common import load_and_preprocess_data, split_data_with_val
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score
import joblib

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_single_output_dnn(input_dim):
    """构建单输出DNN模型"""
    inputs = Input(shape=(input_dim,))
    
    # 共享特征提取层
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # 输出层
    out = Dense(16, activation='relu')(x)
    out = Dense(1)(out)  # 线性输出
    
    model = Model(inputs=inputs, outputs=out)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def DNN():
    # 1. 加载和预处理数据
    X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')
    
    # 2. 划分数据集并标准化
    X_train_scaled, X_val_scaled, X_test_scaled,y_stdv_train_scaled, y_stdv_val_scaled, y_stdv_test_scaled,\
    y_load_train_scaled, y_load_val_scaled, y_load_test_scaled, scalers = split_data_with_val(X, y_stdv, y_load)
    
    # 3. 训练晶粒尺寸标准差模型
    print("训练晶粒尺寸标准差模型...")
    stdv_model = build_single_output_dnn(input_dim=X_train_scaled.shape[1])
    
    callbacks_stdv = [
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights = True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,min_lr=1e-6)
    ]
    
    history_stdv = stdv_model.fit(
        X_train_scaled, y_stdv_train_scaled,
        validation_data=(X_val_scaled, y_stdv_val_scaled),
        epochs=1000,
        batch_size=16,
        callbacks=callbacks_stdv,
        verbose=1
    )
    
    # 4. 训练模具载荷模型
    print("\n训练模具载荷模型...")
    load_model = build_single_output_dnn(input_dim=X_train_scaled.shape[1])
    
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
        verbose=1
    )
    
    # 5. 评估模型
    # 晶粒尺寸标准差模型
    stdv_pred_scaled = stdv_model.predict(X_test_scaled)
    stdv_pred = scalers['scaler_y_stdv'].inverse_transform(stdv_pred_scaled)
    fact_stdv = scalers['scaler_y_stdv'].inverse_transform(y_stdv_test_scaled.reshape(-1, 1))
    
    # 模具载荷模型
    load_pred_scaled = load_model.predict(X_test_scaled)
    load_pred = scalers['scaler_y_load'].inverse_transform(load_pred_scaled)
    fact_load = scalers['scaler_y_load'].inverse_transform(y_load_test_scaled.reshape(-1, 1))
    
    # 计算R²分数
    stdv_r2 = r2_score(fact_stdv, stdv_pred)
    load_r2 = r2_score(fact_load, load_pred)
    
    # 6. 保存模型和标准化器
    os.makedirs("../../data/models/DNN_SingleOutput", exist_ok=True)
    stdv_model.save('../../data/models/DNN_SingleOutput/stdv_model.keras')
    load_model.save('../../data/models/DNN_SingleOutput/load_model.keras')
    joblib.dump(scalers, '../../data/models/DNN_SingleOutput/scalers.pkl')
    
    # 7. 输出结果
    print("\n=== 晶粒尺寸标准差模型 ===")
    print(f"R²分数: {stdv_r2:.4f}")
    print("\n实际值 vs. 预测值(前5行):")
    for i in range(5):
        print(f"实际值: {fact_stdv[i][0]:.4f}, 预测值: {stdv_pred[i][0]:.4f}")
    
    print("\n=== 模具载荷模型 ===")
    print(f"R²分数: {load_r2:.4f}")
    print("\n实际值 vs. 预测值(前5行):")
    for i in range(5):
        print(f"实际值: {fact_load[i][0]:.4f}, 预测值: {load_pred[i][0]:.4f}")

if __name__ == "__main__":
    DNN()