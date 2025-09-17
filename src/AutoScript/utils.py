import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import product
from pyDOE import lhs

def LHSSampleGenerate(n_samples:int, param_ranges:dict,sample_save_path:str):

    # 阶段1：生成拉丁超立方采样样本
    def generate_lhs_samples(n_samples, ranges):  # 修正：添加 n_samples 参数
        """生成拉丁超立方采样输入参数"""
        samples = lhs(len(ranges), samples=n_samples)  # 生成 0-1 之间的 LHS 样本
        scaled_samples = np.zeros_like(samples)
        for i, (key, (low, high)) in enumerate(ranges.items()):
            scaled_samples[:, i] = np.round(samples[:, i] * (high - low) + low).astype(int)
        return pd.DataFrame(scaled_samples, columns=ranges.keys())

    # 生成初始样本
    df = generate_lhs_samples(n_samples, param_ranges)

    # for col in df.columns:
    #     df[col] = df[col].round(1)  # 保留小数点后四位 DEFORM要求

    # 阶段2：添加边界样本（8个角点）
    # 2×2×2
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    boundary_combinations = list(product(*low_high))
    boundary_df = pd.DataFrame(boundary_combinations, columns=param_ranges.keys())
    for col in boundary_df.columns:
        boundary_df[col] = boundary_df[col].astype(int)

    # 合并样本并去重
    df = pd.concat([df, boundary_df], ignore_index=True).drop_duplicates()

    # 保存并输出结果
    SaveAndprint(boundary_df, df, 'lhs',sample_save_path)

def FullSampleGenerate(param_ranges:dict,sample_save_path:str):
    # 阶段1：全因子采样生成初始样本
    def generate_full_factorial_samples(ranges, levels=5):
        """生成全因子采样输入参数"""
        # 为每个参数生成等间隔的水平值
        param_levels = {}
        for key, (low, high) in ranges.items():
            param_levels[key] = np.round(np.linspace(low, high, levels)).astype(int)
    
        # 生成所有参数组合
        combinations = list(product(*param_levels.values()))
        df = pd.DataFrame(combinations, columns=ranges.keys())
        return df
 
    # 生成初始样本（默认每个参数10个水平）
    df = generate_full_factorial_samples(param_ranges, levels=10)

    # for col in df.columns:
    #     df[col] = df[col].round(4)  # 保留小数点后四位 DEFORM要求
 
    # 阶段2：添加边界样本（8个角点）
    low_high = [[low, high] for (low, high) in param_ranges.values()]
    boundary_combinations = list(product(*low_high))
    boundary_df = pd.DataFrame(boundary_combinations, columns=param_ranges.keys())
    for col in boundary_df.columns:
        boundary_df[col] = boundary_df[col].astype(int)
 
    # 合并样本并去重
    df = pd.concat([df, boundary_df], ignore_index=True).drop_duplicates()

    # 保存并输出结果
    SaveAndprint(boundary_df, df, 'fullfactorial',sample_save_path)
 

# df：生成的全因子采样数据
# boundary_df：边界样本数据
# extype：样本类型（如 'fullfactorial'）
def SaveAndprint(boundary_df,df,extype,save_path):
        # 可视化（保持与LHS相同的图表结构）
    plt.figure(figsize=(15, 5))
 
    # 子图1：参数空间分布
    plt.subplot(1, 3, 1)
    plt.scatter(
        df['workpiece_temp'],  # X轴：工件温度
        df['mold_temp'],       # Y轴：模具温度
        c=df['forging_speed'], # 颜色：锻造速度（连续值）
        cmap='viridis',        # 颜色映射表
        s=30                   # 点的大小
    )
    plt.xlabel('Workpiece Temperature (℃)')
    plt.ylabel('Mold Temperature (℃)')
    plt.colorbar(label='Forging Speed (mm/s)')
    plt.title('Parameter Space Distribution')
 
    # 子图2：单参数分布直方图
    plt.subplot(1, 3, 2)
    for col in df.columns:
        plt.hist(df[col], bins=20, alpha=0.5, label=col)
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Individual Parameter Distributions')
 
    # 子图3：边界样本高亮
    plt.subplot(1, 3, 3)
    plt.scatter(df['workpiece_temp'], df['mold_temp'], 
                c='blue', s=20, alpha=0.6, label='Normal Samples')
    plt.scatter(boundary_df['workpiece_temp'], boundary_df['mold_temp'],
                c='red', s=50, marker='*', label='Boundary Samples')
    plt.xlabel('Workpiece Temperature (℃)')
    plt.ylabel('Mold Temperature (℃)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    plt.title('Boundary Samples Highlight')
    plt.grid(False)
 
    plt.tight_layout()
    plt.show()
 
    # 数据保存 添加序号
    df_with_id = df.copy()
    df_with_id.insert(0, 'ID', range(1, len(df_with_id) + 1))
    df_with_id.to_csv(f'{save_path}/IN{extype}.txt', sep='\t', index=False, header=False)

    print(f"采样数据已保存至 {save_path}IN{extype}.txt 总计{len(df)} 个样本")
    print("输入参数统计信息：")
    print(df.describe())