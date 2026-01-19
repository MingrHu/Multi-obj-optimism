from common import evaluate_model
from DNN import dnn_run
from KrigingGPR import kriging_fun
from Polynomial import prg_fun
from SVR import svr_fun
from RandomForest import rf_run
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换工作目录
os.chdir(script_dir)
output_dir = "../../data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "train_history_result.txt")


def main():

    train_times = 20
    models_func = [RunDNN,RunKriging,RunPR,RunSVR,RunRF]

    for func in models_func:
        (mean_stdv_r2,mean_stdv_nmae,max_stdv_time,min_stdv_time,sum_stdv_time,
                mean_load_r2,mean_load_nmae,max_load_time,min_load_time,sum_load_time) = func(train_times)
        
        stdv_score = evaluate_model(mean_stdv_r2,sum_stdv_time / train_times,min_stdv_time,max_stdv_time,0.9,0.1)
        load_score = evaluate_model(mean_load_r2,sum_load_time / train_times,min_load_time,max_load_time,0.9,0.1)

        lines = [
            f"晶粒尺寸INFO: r2均值: {mean_stdv_r2 / train_times:.4f}, nmae均值: {mean_stdv_nmae / train_times:.4f}",
            f"晶粒尺寸INFO: 最大时长: {max_stdv_time:.4f}, 最小时长: {min_stdv_time:.4f}, 平均训练时长: {sum_stdv_time / train_times:.4f}",
            f"模具载荷INFO: r2均值: {mean_load_r2 / train_times:.4f}, nmae均值: {mean_load_nmae / train_times:.4f}",
            f"模具载荷INFO: 最大时长: {max_load_time:.4f}, 最小时长: {min_load_time:.4f}, 平均训练时长: {sum_load_time / train_times:.4f}"
        ]
        with open(output_path,"a",encoding="utf-8") as f:
            for line in lines:
                f.write(line + os.linesep)  
            f.write(f"Score of stdv_model is:{stdv_score}\n")
            f.write(f"Score of load_model is:{load_score}\n\n")

if __name__ == "__main__":
    main()
        

