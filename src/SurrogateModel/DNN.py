import os
import numpy as np
from common import (load_and_preprocess_data, split_data_with_val,
                    normal_max_absolute_error,build_single_output_dnn,save_model,Time)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score

def dnn_run(file:str,vars_out:list[str],n_var:int,model_par:list[str] = []):
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)

    # 2. 划分数据集并标准化
    (X_train_scaled, X_val_scaled, X_test_scaled,
    Y_train_scaled_list, Y_val_scaled_list, Y_test_scaled_list,
    scalers)= split_data_with_val(X, Y)            

    for idx in range(len(Y_train_scaled_list)):
        # 简单三层感知机
        cur_model = build_single_output_dnn(X_train_scaled.shape[1])
        callbacks_stdv = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights = True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,min_lr=1e-6)
        ]
        y_train_scaled = Y_train_scaled_list[idx]
        y_val_scaled = Y_val_scaled_list[idx]
        # 训练模型
        history_stdv = cur_model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=1000,
            batch_size=16,
            callbacks=callbacks_stdv,
            # 1显示训练过程
            verbose=1 # type: ignore
        )

        pred_scaled = cur_model.predict(X_test_scaled)
        test_scaled = Y_test_scaled_list[idx]
        # 调用标准化器
        pred = scalers[f'scaler_y_{idx}'].inverse_transform(pred_scaled.reshape(-1, 1)) # type: ignore
        fact = scalers[f'scaler_y_{idx}'].inverse_transform(test_scaled.reshape(-1, 1))
    
        # 计算相关指标
        r2 = r2_score(fact, pred)
        nmae = normal_max_absolute_error(fact, pred)

        # 保存并打印当前结果
        save_model(f"{vars_out[idx + n_var]}",cur_model,r2,fact,pred,scalers,"DNN")    

# if __name__ == "__main__":
#     vars_out = ["1","2","3","4","5","6","7","res1","res2","res3"]
#     file = '/Users/hmr/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'
#     dnn_run(file,vars_out,7)




