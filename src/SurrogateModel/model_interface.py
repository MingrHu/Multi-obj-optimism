from DNN import dnn_run
from KrigingGPR import kriging_fun
from Polynomial import prg_fun
from SVR import svr_fun
from RandomForest import rf_run
#  @brief  代理模型类
#  @return None
#  @author Hu Mingrui
#  @date   2025/11/27
#  @param  data_file        数据集文件 请输入绝对路径 例如c//users//hmr//desktop//file.txt
#  @param  vars_out         是为了标识不同的列取的名字 同时用于控制输入输出位置以7输入和3输出为例：
#  例如下 其中1至7是输入参数 可以认为是工艺参数名称 后面的三个数数优化的目标值名称
#  vars_out = ["1","2","3","4","5","6","7","res1","res2","res3"]
#  @about  基础的代理模型类 对外提供的统一的代理模型接口
class Doe_surrogateModel:
    def __init__(self, 
                 data_file: str,  # 数据集文件
                 vars_out:list[str],  # 数据集的输入参数和输出目标值名
                 n_vars:int, # 自变量X 输入参数个数
                ):
        self.file = data_file
        self.vars_out = vars_out
        self.n = n_vars
        self.model = [kriging_fun,dnn_run,prg_fun,svr_fun,rf_run]
    # which_model:取值0-4 分别对应如下
    # [kriging_fun,dnn_run,prg_fun,svr_fun,rf_run]
    def train_save_model(self,which_model:int,model_par:list[str] = []):
        self.model[which_model](self.file,self.vars_out,self.n,model_par)

if __name__ == "__main__":
    vars_out = ["1", "2", "3", "4", "5", "6", "7","8","9","10",
                "11","12","13","14","15","16","17","18","19","20", "res1", "res2", "res3"]
    file = '/Users/bytedance/Desktop/Multi-obj-optimism/data/TEST/simulated.txt'
    doe_s = Doe_surrogateModel(file,vars_out,20)
    doe_s.train_save_model(2)


