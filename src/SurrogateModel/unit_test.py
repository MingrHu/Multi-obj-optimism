import pandas as pd
import numpy as np
import joblib
import os
from common import load_and_preprocess_data, split_data_with_val
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 1. 数据加载与预处理
X, y_stdv, y_load = load_and_preprocess_data('../../data/RES-108.txt')

# 3. 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 目标变量标准化
scaler_y_stdv = StandardScaler()
y_stdv_scaled = scaler_y_stdv.fit_transform(y_stdv)

scaler_y_load = StandardScaler()
y_load_scaled = scaler_y_load.fit_transform(y_load)

# 4. 数据划分
X_train, X_temp, y_train_stdv, y_temp_stdv, y_train_load, y_temp_load = train_test_split(
    X_scaled, y_stdv_scaled, y_load_scaled, test_size=0.2, random_state=42
)
X_val, X_test, y_val_stdv, y_test_stdv, y_val_load, y_test_load = train_test_split(
    X_temp, y_temp_stdv, y_temp_load, test_size=0.5, random_state=42
)

# 5. 定义多输出DNN模型构建函数
def build_multi_output_dnn(input_dim):
    # 输入层
    inputs = Input(shape=(input_dim,))
    
    # 共享特征提取层
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # 创建两个独立输出分支
    # 分支1: 预测晶粒尺寸标准差
    out1 = Dense(16, activation='relu')(x)
    out1 = Dense(1, name='grain_size')(out1)  # 线性输出
    
    # 分支2: 预测模具载荷
    out2 = Dense(16, activation='relu')(x)
    out2 = Dense(1, name='mold_load')(out2)  # 线性输出
    
    # 构建多输出模型
    model = Model(inputs=inputs, outputs=[out1, out2])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'grain_size': 'mse', 'mold_load': 'mse'},
        metrics={'grain_size': 'mae', 'mold_load': 'mae'},
        loss_weights=[0.5, 0.5]  # 平衡两个任务的损失
    )
    return model

# 6. 创建并训练多输出模型
multi_output_model = build_multi_output_dnn(input_dim=X_train.shape[1])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# 训练模型 - 同时使用两个目标
history = multi_output_model.fit(
    X_train,
    {'grain_size': y_train_stdv, 'mold_load': y_train_load},  # 双目标训练数据
    validation_data=(
        X_val,
        {'grain_size': y_val_stdv, 'mold_load': y_val_load}  # 双目标验证数据
    ),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7. 预测
# 多输出模型的预测结果是两个数组
predictions = multi_output_model.predict(X_test)
stdv_pred_scaled, load_pred_scaled = predictions

# 反标准化预测值
stdv_pred = scaler_y_stdv.inverse_transform(stdv_pred_scaled)
load_pred = scaler_y_load.inverse_transform(load_pred_scaled)

# 反标准化实际值
fact_stdv = scaler_y_stdv.inverse_transform(y_test_stdv)
fact_load = scaler_y_load.inverse_transform(y_test_load)

# 计算R²分数
stdv_r2 = r2_score(fact_stdv, stdv_pred)
load_r2 = r2_score(fact_load, load_pred)

# # 8. 保存模型和标准化器
# os.makedirs("models/DNN_Optimized", exist_ok=True)
# multi_output_model.save('models/DNN_Optimized/multi_output_model.keras')
# joblib.dump(scaler_X, 'models/DNN_Optimized/scaler_X.pkl')
# joblib.dump(scaler_y_stdv, 'models/DNN_Optimized/scaler_y_stdv.pkl')
# joblib.dump(scaler_y_load, 'models/DNN_Optimized/scaler_y_load.pkl')

# 9. 输出结果
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