# ********************************************
#  创建时间： 2025/09/23
#  名    称： 代理模型公共库
#  版    本： V1.0
#  @author    Hu Mingrui
#  说    明： 负责其余代理模型模块调用方法
# *******************************************
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib,os,time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

#####################################数据处理函数块############################################
# 数据加载与预处理函数
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
                  'STDV_grainSize','Die_Load']
    
    # 特征与目标变量
    X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']].values
    y_stdv = df['STDV_grainSize'].values.reshape(-1, 1)
    y_load = df['Die_Load'].values.reshape(-1, 1)
    
    return X, y_stdv, y_load

# 划分数据集 不要验证集
def split_data_without_val(X, y_stdv, y_load, test_size=0.2,random_state=42):
    # 划分训练集和测试集
    X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
        X, y_stdv, y_load, test_size=test_size,random_state=random_state
    )
 
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
 
    # 目标值1标准化（晶粒尺寸标准差）
    scaler_y_stdv = StandardScaler()
    y_stdv_train_scaled = scaler_y_stdv.fit_transform(y_stdv_train).ravel()  
    y_stdv_test_scaled = scaler_y_stdv.transform(y_stdv_test).ravel()       
 
    # 目标值2标准化（模具载荷）
    scaler_y_load = StandardScaler()
    y_load_train_scaled = scaler_y_load.fit_transform(y_load_train).ravel()  
    y_load_test_scaled = scaler_y_load.transform(y_load_test).ravel()       
 
    # 返回标准化后的数据和标准化器
    return (
        X_train_scaled, X_test_scaled,
        y_stdv_train_scaled, y_stdv_test_scaled,
        y_load_train_scaled, y_load_test_scaled,
        {
            'scaler_X': scaler_X,
            'scaler_y_stdv': scaler_y_stdv,
            'scaler_y_load': scaler_y_load
        }
    )

# 划分数据集 需要验证集
def split_data_with_val(X, y_stdv, y_load, test_size=0.2, val_size=0.25, random_state=42):
    # 特征与目标变量
    X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
        X, y_stdv, y_load, test_size=test_size, random_state=random_state
    )
    
    # 第一步：划分训练集和临时集(测试+验证)
    X_train, X_temp, y_stdv_train, y_stdv_temp, y_load_train, y_load_temp = train_test_split(
        X, y_stdv, y_load, test_size=test_size + val_size, random_state=random_state
    )
    
    # 第二步：在临时集中划分验证集和测试集
    X_val, X_test, y_stdv_val, y_stdv_test, y_load_val, y_load_test = train_test_split(
        X_temp, y_stdv_temp, y_load_temp, 
        test_size=val_size/(test_size + val_size), 
        random_state=random_state
    )
    
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 目标值标准化（晶粒尺寸标准差）
    scaler_y_stdv = StandardScaler()
    y_stdv_train_scaled = scaler_y_stdv.fit_transform(y_stdv_train).ravel()
    y_stdv_val_scaled = scaler_y_stdv.transform(y_stdv_val).ravel()
    y_stdv_test_scaled = scaler_y_stdv.transform(y_stdv_test).ravel()
    
    # 目标值标准化（模具载荷）
    scaler_y_load = StandardScaler()
    y_load_train_scaled = scaler_y_load.fit_transform(y_load_train).ravel()
    y_load_val_scaled = scaler_y_load.transform(y_load_val).ravel()
    y_load_test_scaled = scaler_y_load.transform(y_load_test).ravel()
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_stdv_train_scaled, y_stdv_val_scaled, y_stdv_test_scaled,
        y_load_train_scaled, y_load_val_scaled, y_load_test_scaled,
        {
            'scaler_X': scaler_X,
            'scaler_y_stdv': scaler_y_stdv,
            'scaler_y_load': scaler_y_load
        }
    )

# 计算NMAE函数
def normal_max_absolute_error(y_true, y_pred):
    """
    计算 Normal Maximum Absolute Error (NMAE):
    NMAE = max(|y_true - y_pred|) / RMSE(y_true - y_pred)
    """
    errors = np.abs(y_true - y_pred)
    max_ae = np.max(errors)  # 分子：最大绝对误差
    rmse = np.sqrt(np.mean(errors ** 2))  # 分母：预测误差的 RMSE
    
    if rmse == 0:
        return np.inf if max_ae > 0 else 0.0  # 避免除以 0
    return max_ae / rmse

# 训练的模型类型 最佳模型 最佳决定系数 实际值和预测值 代理模型名称
def save_best_model(model_type,best_model,best_r2,best_fact,best_pred,scalers,model_name):
    # 训练结束后，保存最佳模型
    os.makedirs(f"../../data/models/{model_name}", exist_ok=True)
    if best_model is not None:
        if model_name == 'DNN':
            best_model.save(f'../../data/models/{model_name}/best_{model_type}_model.keras')
            joblib.dump(scalers, f'../../data/models/{model_name}/{model_type}_scalers.pkl')
        else:
            joblib.dump(best_model,f'../../data/models/{model_name}/best_{model_type}_model.pkl')
            joblib.dump(scalers, f'../../data/models/{model_name}/{model_type}_scalers.pkl')
        print("\n=== 最佳模型已保存 ===")
        print(f"最高 R²分数: {best_r2:.4f}")
        
        # 打印最佳模型的实际值 vs. 预测值（前5行）
        print("\n最佳模型预测结果(前5行):")
        for j in range(5):
            print(f"实际值: {best_fact[j][0]:.4f}, 预测值: {best_pred[j][0]:.4f}")
    else:
        print("警告：未找到有效模型！")
#####################################数据处理函数块############################################



#####################################工具辅助函数模块############################################
import time

class Time:
    """记录代码块或函数执行时间的工具类（支持手动和自动模式）"""

    def __init__(self, name=None):
        self.name = name  # 可选：标记计时名称
        self.start_time = None
        self.end_time = None
        self.duration = None  # 存储计算后的时长（秒）

    def start(self):
        """手动开始计时"""
        self.start_time = time.perf_counter()
        return self  # 返回自身，支持链式调用

    def stop(self):
        """手动停止计时，并计算时长"""
        if self.start_time is None:
            raise RuntimeError("请先调用 start() 开始计时！")
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self  # 返回自身，支持链式调用

    def get_duration(self, unit="s"):
        """
        获取计时时长
        :param unit: 时间单位，可选 "s"（秒）、"ms"（毫秒）、"us"（微秒）
        :return: 转换后的时长
        """
        if self.duration is None:
            raise RuntimeError("请先调用 stop() 停止计时！")
        if unit == "s":
            return self.duration
        elif unit == "ms":
            return self.duration * 1000
        elif unit == "us":
            return self.duration * 1_000_000
        else:
            raise ValueError("不支持的单位，请选择 's'、'ms' 或 'us'")

    def print_duration(self, unit="ms"):
        """打印计时结果"""
        duration = self.get_duration(unit)
        unit_name = {"s": "秒", "ms": "毫秒", "us": "微秒"}[unit]
        if self.name:
            print(f"[{self.name}] 耗时: {duration:.4f} {unit_name}")
        else:
            print(f"代码块耗时: {duration:.4f} {unit_name}")

    # 以下保持原有的上下文管理器和装饰器功能
    def __enter__(self):
        """上下文管理器入口，记录开始时间"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，记录结束时间并打印耗时"""
        self.stop()
        self.print_duration(unit="ms")

    def __call__(self, func):
        """装饰器：记录被装饰函数的执行时间"""
        def wrapper(*args, **kwargs):
            with self:
                if self.name is None:
                    self.name = func.__name__  # 默认用函数名标记
                return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def record(func=None, name=None):
        """
        静态方法装饰器（支持带参数的装饰器）
        示例：
            @Time.record()
            def foo(): ...

            @Time.record(name="自定义名称")
            def bar(): ...
        """
        if func is None:
            return lambda f: Time(name=name)(f)
        return Time(name=name)(func)

def evaluate_model(R2, T, T_min, T_max, w1=0.5, w2=0.5):
    # 标准化精度
    f_R2 = min(1.0, R2)
    # 标准化时间
    g_T = 1 - (T - T_min) / (T_max - T_min) if T_max != T_min else 1.0
    # 综合得分
    score = w1 * f_R2 + w2 * g_T
    return score


#####################################工具辅助函数模块############################################




#####################################各类代理模型定义部分######################################
# DNN的模型定义
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