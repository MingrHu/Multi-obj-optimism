import os
import sys
import numpy as np
import joblib
import gymnasium as gym
from gymnasium import spaces

# 添加上级目录以导入 SurrogateModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ForgingEnv(gym.Env):
    """
    基于 Gymnasium 的锻造工艺优化环境
    """
    def __init__(self, model_family="PRG"):
        super(ForgingEnv, self).__init__()
        
        self.model_dir = f"../../data/models/{model_family}"
        self.objective_names = ["grain", "load"]
        
        # 1. 加载标准化器和模型
        self._load_resources()
        
        # 2. 定义动作空间 (3个工艺参数的归一化调整量 [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 3. 定义状态空间 (3个工艺参数的实际值)
        # 工件温度(875-965), 模具温度(300-700), 上模速度(10-50)
        self.low_bound = np.array([875.0, 300.0, 10.0], dtype=np.float32)
        self.high_bound = np.array([965.0, 700.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_bound, high=self.high_bound, dtype=np.float32)
        
        self.current_state = None
        self.weights = np.array([0.5, 0.5]) # 默认权重

    def _load_resources(self):
        # 加载 Scaler (假设 grain_scalers 包含所有信息)
        scaler_path = os.path.join(self.model_dir, f"{self.objective_names[0]}_scalers.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器未找到: {scaler_path}")
        
        self.scalers = joblib.load(scaler_path)
        self.scaler_X = self.scalers["scaler_X"]
        
        # 加载模型
        self.models = {}
        for name in self.objective_names:
            pkl_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            if os.path.exists(pkl_path):
                self.models[name] = joblib.load(pkl_path)
            else:
                # 尝试加载 keras 模型
                from keras.models import load_model
                keras_path = os.path.join(self.model_dir, f"{name}_model.keras")
                self.models[name] = load_model(keras_path)

    def reset(self, seed=None, options=None): # type: ignore
        super().reset(seed=seed)
        # 随机初始化状态
        self.current_state = self.np_random.uniform(self.low_bound, self.high_bound).astype(np.float32)
        
        # 如果提供了权重，则更新
        if options and "weights" in options:
            self.weights = np.array(options["weights"])
            
        return self.current_state, {}

    def step(self, action):
        # 1. 更新状态 (动作是步长调整)
        step_size = (self.high_bound - self.low_bound) * 0.05 # 每次最大调整 5% 范围
        delta = action * step_size
        next_state = np.clip(self.current_state + delta, self.low_bound, self.high_bound)
        self.current_state = next_state
        
        # 2. 预测目标值
        objectives = self._predict(next_state)
        
        # 3. 计算奖励 (归一化后的加权和，越小越好 -> 负值奖励)
        # 归一化参考: grain~[10,50], load~[200k, 400k]
        norm_grain = (objectives["grain"] - 10) / 40.0
        norm_load = (objectives["load"] - 200000) / 200000.0
        
        reward = -(self.weights[0] * norm_grain + self.weights[1] * norm_load)
        
        # 约束惩罚
        if objectives["grain"] > 30 or objectives["load"] > 330000:
            reward -= 5.0
            
        # 4. 结束条件 (这里设为 False，由外部控制步数)
        terminated = False
        truncated = False
        
        return next_state, reward, terminated, truncated, objectives

    def _predict(self, x):
        x_scaled = self.scaler_X.transform(x.reshape(1, -1))
        results = {}
        for i, name in enumerate(self.objective_names):
            y_pred_scaled = self.models[name].predict(x_scaled)
            if isinstance(y_pred_scaled, np.ndarray):
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            scaler_y = self.scalers[f"scaler_y_{i}"]
            results[name] = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        return results