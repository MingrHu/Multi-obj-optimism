from .common import (load_and_preprocess_data,split_data_without_val,
                    save_model)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from typing import Any

from .hyperparameters import normalize_model_params

def rf_run(file: str, vars_out: list[str], n_var: int,
           model_par: dict[str, Any] | None = None):
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)
    # 2. 划分数据集并标准化
    (X_train_scaled, X_test_scaled,
    Y_train_scaled_list, Y_test_scaled_list,
    scalers) = split_data_without_val(X, Y)            

    params = normalize_model_params("RF", model_par)
    for idx in range(len(Y_train_scaled_list)):
        cur_model = RandomForestRegressor(**params)
        # 训练模型
        cur_model.fit(X_train_scaled, Y_train_scaled_list[idx])
        pred_scaled = cur_model.predict(X_test_scaled)
        test_scaled = Y_test_scaled_list[idx]
        # 调用标准化器
        pred = scalers[f'scaler_y_{idx}'].inverse_transform(pred_scaled.reshape(-1, 1)) # type: ignore
        fact = scalers[f'scaler_y_{idx}'].inverse_transform(test_scaled.reshape(-1, 1))
    
        # 计算相关指标
        r2 = r2_score(fact, pred)
        # 保存并打印当前结果
        save_model(f"{vars_out[idx + n_var]}",cur_model,r2,fact,pred,scalers,"RF")
