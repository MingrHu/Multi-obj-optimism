from .common import (load_and_preprocess_data, split_data_with_val,
                    build_single_output_dnn,save_model)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score
from typing import Any

from .hyperparameters import normalize_model_params

def dnn_run(file: str, vars_out: list[str], n_var: int,
            model_par: dict[str, Any] | None = None):
    # 1. 加载数据
    X, Y = load_and_preprocess_data(file,vars_out,n_var)

    # 2. 划分数据集并标准化
    (X_train_scaled, X_val_scaled, X_test_scaled,
    Y_train_scaled_list, Y_val_scaled_list, Y_test_scaled_list,
    scalers)= split_data_with_val(X, Y)            

    params = normalize_model_params("DNN", model_par)
    for idx in range(len(Y_train_scaled_list)):
        # 简单三层感知机
        cur_model = build_single_output_dnn(
            X_train_scaled.shape[1],
            hidden_layer_1=params["hidden_layer_1"],
            hidden_layer_2=params["hidden_layer_2"],
            hidden_layer_3=params["hidden_layer_3"],
            activation=params["activation"],
            dropout_1=params["dropout_1"],
            dropout_2=params["dropout_2"],
            batch_normalization=params["batch_normalization"],
            learning_rate=params["learning_rate"],
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=params["patience"], restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=params["reduce_lr_factor"],
                patience=params["reduce_lr_patience"],
                min_lr=params["min_lr"],
            ),
        ]
        y_train_scaled = Y_train_scaled_list[idx]
        y_val_scaled = Y_val_scaled_list[idx]
        # 训练模型
        cur_model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            # 1显示训练过程
            verbose=params["verbose"] # type: ignore
        )

        pred_scaled = cur_model.predict(X_test_scaled)
        test_scaled = Y_test_scaled_list[idx]
        # 调用标准化器
        pred = scalers[f'scaler_y_{idx}'].inverse_transform(pred_scaled.reshape(-1, 1)) # type: ignore
        fact = scalers[f'scaler_y_{idx}'].inverse_transform(test_scaled.reshape(-1, 1))
    
        # 计算相关指标
        r2 = r2_score(fact, pred)
        # 保存并打印当前结果
        save_model(f"{vars_out[idx + n_var]}",cur_model,r2,fact,pred,scalers,"DNN")    

