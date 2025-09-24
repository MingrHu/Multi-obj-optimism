from common import evaluate_model
from DNN import RunDNN
from KrigingGPR import RunKriging
from Polynomial import RunPR
from SVR import RunSVR
from RandomForest import RunRF
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
        lines = [
            f"晶粒尺寸INFO: r2均值: {mean_stdv_r2 / train_times:.4f}, nmae均值: {mean_stdv_nmae / train_times:.4f}",
            f"晶粒尺寸INFO: 最大时长: {max_stdv_time:.4f}, 最小时长: {min_stdv_time:.4f}, 平均训练时长: {sum_stdv_time / train_times:.4f}",
            f"模具载荷INFO: r2均值: {mean_load_r2 / train_times:.4f}, nmae均值: {mean_load_nmae / train_times:.4f}",
            f"模具载荷INFO: 最大时长: {max_load_time:.4f}, 最小时长: {min_load_time:.4f}, 平均训练时长: {sum_load_time / train_times:.4f}"
        ]
        with open(output_path,"a",encoding="utf-8") as f:
            for line in lines:
                f.write(line + os.linesep)  

if __name__ == "__main__":
    # main()
    data = [[0.7414,13145.083,4702.4117,24555.6449],
            [0.6826,12561.505,4464.9423,19499.3428],

            [0.7448,546.7178,424.8733,879.5209],
            [0.6465,556.0638,387.7369,913.4037],

            [0.7593,7.9518,3.5886,18.5514],
            [0.7661,4.7113,3.5982,9.0750],

            [0.8883,4.6057,3.6196,6.4748],
            [0.7983,4.4182,3.2798,5.2710],

            [0.8086,186.1393,165.7085,216.6970],
            [0.6467,69.5075,59.3589,81.7606]
            ]
    
    for i,v in enumerate(data):
        if i % 2 == 0:
            temp = evaluate_model(v[0],v[1],v[2],v[3],0.9,0.1)
            with open(output_path,"a",encoding="utf-8") as f:
                f.write(f"Score of stdv_model is:{temp}\n")
        else:
            temp = evaluate_model(v[0],v[1],v[2],v[3],0.9,0.1)
            with open(output_path,"a",encoding="utf-8") as f:
                f.write(f"Score of load_model is:{temp}\n")
        

