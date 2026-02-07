import pandas as pd

# 读取文件（假设列是以空格/制表符分隔）
df = pd.read_csv("simulated.txt", sep="\t", header=None)  # \s+ 匹配任意空白字符
df = df.drop(columns=[0])  # 删除第 0 列（第一列）
df.to_csv('simulated.txt', sep='\t', index=False, header=False,float_format = "%.2f")