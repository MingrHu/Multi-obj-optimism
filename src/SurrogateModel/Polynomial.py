import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录
os.chdir(script_dir)

def main():
    # 1.数据装载
    df = pd.read_csv('../../data/RES-108.txt', sep='\t', header=None)
    df.columns = ['Index', 'Workpiece_Temp', 'Die_Temp', 'Forming_Speed', 
                  'STDV_grainSize', 'Max_grainSize','Avg_grainSize','Die_Load']

    # 2. 特征与目标变量
    X = df[['Workpiece_Temp', 'Die_Temp', 'Forming_Speed']]
    y_stdv = df['STDV_grainSize']
    y_load = df['Die_Load']

    # 3. 划分数据集（80%训练，20%测试）
    X_train, X_test, y_stdv_train, y_stdv_test, y_load_train, y_load_test = train_test_split(
        X, y_stdv, y_load, test_size=0.2, random_state=42
    )

    # 4. 定义多项式回归模型（这里使用2次多项式，可以根据需要调整）
    degree = 2  # 多项式阶数
    # def assesment(Max:int):
    #     res = [[] for _ in range(Max)]
    #     for num in range(1,Max + 1):
    #         cur_load_model = make_pipeline(PolynomialFeatures(num, include_bias=False), 
    #                           LinearRegression())
    #         cur_stdv_model = make_pipeline(PolynomialFeatures(num, include_bias=False), 
    #                            LinearRegression())
    #         cur_stdv_model.fit(X_train,y_stdv_train)
    #         cur_load_model.fit(X_train, y_load_train)
    #         res[num - 1].append(cur_stdv_model)
    #         res[num - 1].append(cur_load_model)
    #     pos1,pos2 = -1,-1
    #     for i in range(len(res)):
    #         test_stdv_model = res[i][0]
    #         test_load_model = res[i][1]
    #         stdv_pred = test_stdv_model.predict(X_test)
    #         load_pred = test_load_model.predict(X_test)
    #         R2_stdv,R2_load = r2_score(y_stdv_test,stdv_pred),r2_score(y_load_test, load_pred)
    #         print(f"最高次系数为{i+1}  晶粒标准差r2 = {R2_stdv},载荷模型r2 = {R2_load}")

    # 晶粒尺寸标准差模型
    stdv_model = make_pipeline(PolynomialFeatures(degree, include_bias=False), 
                            LinearRegression())
    stdv_model.fit(X_train, y_stdv_train)
    stdv_pred = stdv_model.predict(X_test)
    stdv_r2 = r2_score(y_stdv_test, stdv_pred)

    # 模具载荷模型
    load_model = make_pipeline(PolynomialFeatures(degree, include_bias=False), 
                            LinearRegression())
    load_model.fit(X_train, y_load_train)
    load_pred = load_model.predict(X_test)
    load_r2 = r2_score(y_load_test, load_pred)

    # 5. 保存模型
    os.makedirs("../../data/models/PRG", exist_ok=True)  
    joblib.dump(stdv_model, '../../data/models/PRG/stdv_model.pkl')
    joblib.dump(load_model, '../../data/models/PRG/die_load_model.pkl')

    # 6. 输出结果

    print("\n=== 晶粒尺寸标准差模型 ===")
    print(f"多项式阶数: {degree}")
    print(f"R²分数: {stdv_r2:.4f}")
    print("\n实际值 vs. 预测值(前5行):")
    print(pd.DataFrame({'实际值': y_stdv_test.head(5), '预测值': stdv_pred[:5]}))

    print("\n=== 模具载荷模型N ===")
    print(f"多项式阶数: {degree}")
    print(f"R²分数: {load_r2:.4f}")
    print("\n实际值 vs. 预测值(前5行):")
    print(pd.DataFrame({'实际值': y_load_test.head(5), '预测值': load_pred[:5]}))

if __name__ == "__main__":
    main()