from common import (load_and_preprocess_data,split_data_without_val,
                    normal_max_absolute_error,save_model,Time)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score

def kriging_fun(file:str,vars_out:list[str],n_var:int,model_par:list[str] = []):
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)

    # 2. 划分数据集并标准化
    (X_train_scaled, X_test_scaled,
    Y_train_scaled_list, Y_test_scaled_list,
    scalers) = split_data_without_val(X, Y)            

    for idx in range(len(Y_train_scaled_list)):
        # 核函数 这也是能够调整的参数的地方
        kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e4))
        cur_model = GaussianProcessRegressor(
            kernel = kernel,
            n_restarts_optimizer = 20,
            alpha = 0.1,
            random_state = 42
        )
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
        save_model(f"{vars_out[idx + n_var]}",cur_model,r2,fact,pred,scalers,"KM")    

# if __name__ == "__main__":
#     vars_out = ["1","2","3","4","5","6","7","res1","res2","res3"]
#     file = '/Users/hmr/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'
#     kriging_fun(file,vars_out,7)