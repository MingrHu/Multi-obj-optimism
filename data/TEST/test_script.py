import numpy as np
import pandas as pd

# 设置随机种子，保证生成的数据可复现
np.random.seed(42)

# 1. 定义参数
n_rows = 4375    # 数据行数
n_features = 20  # 自变量
n_targets = 3   # 因变量

# 2. 生成自变量 X (100行×7列)
# 采用不同的随机分布模拟不同类型的特征（如温度、速度、压力等）
X = np.hstack([
    np.random.normal(loc=50, scale=10, size=(n_rows, n_features // 3)),   # 特征1：正态分布
    np.random.uniform(low=0, high=100, size=(n_rows, n_features // 2)),  # 特征2：均匀分布
    np.random.poisson(lam=20, size=(n_rows, n_features - n_features // 3 - n_features // 2))  # 特征3：泊松分布
])

# 3. 生成因变量 Y (100行×3列)
# 基于自变量线性组合 + 少量噪声生成，保证Y和X有相关性
# 随机生成系数矩阵 (7个特征 × 3个目标)
coef = np.random.uniform(low=0.5, high=2.0, size=(n_features, n_targets))
# 生成Y：X·coef + 偏置 + 噪声

Y = X @ coef + np.array([[10, 20, 30]]) + np.random.normal(loc=0, scale=2, size=(n_rows, n_targets))

# 4. 合并自变量和因变量，得到完整数据集
data = np.hstack([X, Y])

# 5. 构造DataFrame并命名列名
columns = [f'feature_{i+1}' for i in range(n_features)] + [f'target_{i+1}' for i in range(n_targets)]
df = pd.DataFrame(data, columns=columns)

# ------------- 关键修改：强制保留所有数值两位小数 -------------
# 方法1：对整个DataFrame进行四舍五入，保留两位小数（简洁高效，推荐）
df = df.round(2)

df.to_csv('simulated.txt', sep='\t', index=False, header=False)

# 7. 查看数据基本信息
print("数据集形状：", df.shape)
print("\n数据集前5行：")
print(df.head())
print("\n数据统计信息：")
print(df.describe())