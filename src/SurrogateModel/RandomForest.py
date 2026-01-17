import os
from common import (load_and_preprocess_data,split_data_without_val,
                    normal_max_absolute_error,save_model,Time)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def rf_run(file:str,vars_out:list[str],n_var:int,model_par:list[str] = []):
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)

    # 2. 划分数据集并标准化
    (X_train_scaled, X_test_scaled,
    Y_train_scaled_list, Y_test_scaled_list,
    scalers) = split_data_without_val(X, Y)            

    for idx in range(len(Y_train_scaled_list)):
        num = 300
        cur_model = RandomForestRegressor(n_estimators=num, random_state=42)
        # 训练模型
        cur_model.fit(X_train_scaled, Y_train_scaled_list[idx])
        pred_scaled = cur_model.predict(X_test_scaled)
        test_scaled = Y_test_scaled_list[idx]
        # 调用标准化器
        pred = scalers[f'scaler_y_{idx}'].inverse_transform(pred_scaled.reshape(-1, 1)) # type: ignore
        fact = scalers[f'scaler_y_{idx}'].inverse_transform(test_scaled.reshape(-1, 1))
    
        # 计算相关指标
        r2 = r2_score(fact, pred)
        nmae = normal_max_absolute_error(fact, pred)

        # 保存并打印当前结果
        save_model(f"{vars_out[idx + n_var]}",cur_model,r2,fact,pred,scalers,"RF")    

# if __name__ == "__main__":
#     vars_out = ["1","2","3","4","5","6","7","res1","res2","res3"]
#     file = '/Users/hmr/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'
#     rf_run(file,vars_out,7)