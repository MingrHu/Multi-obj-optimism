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


# 数据加载与预处理函数
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
                  'STDV_grainSize', 'Max_grainSize','Avg_grainSize','Die_Load']
    
    # 特征与目标变量
    X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']].values
    y_stdv = df['STDV_grainSize'].values.reshape(-1, 1)
    y_load = df['Die_Load'].values.reshape(-1, 1)
    
    return X, y_stdv, y_load

# 划分数据集 不要验证集
def split_data_without_val(X, y_stdv, y_load, test_size=0.2, random_state=42):
    # 划分训练集和测试集
    X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
        X, y_stdv, y_load, test_size=test_size, random_state=random_state
    )
 
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
 
    # 目标值1标准化（晶粒尺寸标准差）
    scaler_y_stdv = StandardScaler()
    y_stdv_train_scaled = scaler_y_stdv.fit_transform(y_stdv_train.reshape(-1, 1)).ravel()  # 移除 .values
    y_stdv_test_scaled = scaler_y_stdv.transform(y_stdv_test.reshape(-1, 1)).ravel()       # 移除 .values
 
    # 目标值2标准化（模具载荷）
    scaler_y_load = StandardScaler()
    y_load_train_scaled = scaler_y_load.fit_transform(y_load_train.reshape(-1, 1)).ravel()  # 移除 .values
    y_load_test_scaled = scaler_y_load.transform(y_load_test.reshape(-1, 1)).ravel()       # 移除 .values
 
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