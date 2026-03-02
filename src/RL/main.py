import os
import numpy as np
from stable_baselines3 import PPO
from RL_env import ForgingEnv

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def train_and_optimize():
    # 1. 创建环境
    env = ForgingEnv(model_family="PRG")
    
    # 2. 定义不同的权重组合来寻找 Pareto 前沿
    weights_list = [
        [1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.0, 1.0]
    ]
    
    solutions = []
    
    print("开始强化学习优化...")
    
    for w in weights_list:
        print(f"\n正在优化权重: Grain={w[0]}, Load={w[1]}")
        
        # 重置环境权重
        env.reset(options={"weights": w})
        
        # 3. 初始化并训练 PPO 模型
        # MlpPolicy 适用于向量输入
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
        model.learn(total_timesteps=2000)
        
        # 4. 测试并收集结果
        obs, _ = env.reset(options={"weights": w})
        for _ in range(20):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # 记录满足约束的解
            if info["grain"] <= 30 and info["load"] <= 330000:
                solutions.append({
                    "params": obs.copy(),
                    "grain": info["grain"],
                    "load": info["load"]
                })
    
    # 5. 保存结果
    save_solutions(solutions)

def save_solutions(solutions):
    output_file = "../../data/rl_solutions_sb3.txt"
    
    # 简单去重和排序
    unique_sols = []
    seen = set()
    for s in solutions:
        key = tuple(np.round(s["params"], 2))
        if key not in seen:
            seen.add(key)
            unique_sols.append(s)
    
    unique_sols.sort(key=lambda x: x["grain"])
    
    with open(output_file, "w") as f:
        f.write("RL Optimization Results (Stable Baselines3)\n")
        f.write("==================================================\n")
        f.write(f"Total Solutions: {len(unique_sols)}\n\n")
        f.write("Temp(C) | Die(C) | Speed(mm/s) | Grain(um) | Load(N)\n")
        f.write("--------|--------|-------------|-----------|---------\n")
        for s in unique_sols:
            p = s["params"]
            f.write(f"{p[0]:7.1f} | {p[1]:6.1f} | {p[2]:11.1f} | {s['grain']:9.2f} | {s['load']:.0f}\n")
            
    print(f"\n优化完成！结果已保存至 {output_file}")

if __name__ == "__main__":
    train_and_optimize()