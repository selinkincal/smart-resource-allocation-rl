import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from env import ResourceAllocationEnv
from greedy import GreedyAgent
from dqn_agent import DQNAgent
from dp_solver import DPSolver

def evaluate_models(num_episodes=20):
    # تهيئة البيئة بناءً على معايير مشكلة حقيبة الظهر (Knapsack Problem)
    env = ResourceAllocationEnv(max_cpu=100.0, max_ram=100.0, max_steps=50)
    
    # تحديد المسارات المطلقة لضمان الوصول للملفات من أي مكان
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    models_dir = os.path.join(project_root, "saved_models")
    plots_dir = os.path.join(project_root, "plots")
    
    # تهيئة الوكلاء (Greedy, DQN, DP)
    greedy_agent = GreedyAgent(threshold=1.0)
    
    dqn_agent = DQNAgent(state_size=5, action_size=2)
    model_path = os.path.join(models_dir, "dqn_model.pth")
    
    # تحميل النموذج المدرب إذا كان موجوداً
    if os.path.exists(model_path):
        dqn_agent.qnetwork_local.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded trained model from: {model_path}")
    else:
        print("Warning: Trained model not found. DQN will act with random weights!")
        
    dqn_agent.qnetwork_local.eval() 
    
    dp_solver = DPSolver(max_cpu=100.0, max_ram=100.0)
    
    greedy_scores = []
    dqn_scores = []
    dp_scores = []
    
    print("Starting the comprehensive final comparison (Greedy vs DQN vs DP)...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        seed = 42 + episode
        
        # 1. تقييم الخوارزمية الجشعة وجمع المهام للبرمجة الديناميكية
        obs_greedy, _ = env.reset(seed=seed)
        score_greedy = 0
        terminated = False
        tasks_this_episode = [] 
        
        while not terminated:
            # تسجيل المهام لتحليلها بواسطة DP (Oracle)
            tasks_this_episode.append((env.task_cpu, env.task_ram, env.task_reward))
            
            action = greedy_agent.select_action(obs_greedy)
            obs_greedy, reward, terminated, _, _ = env.step(action)
            score_greedy += reward
        greedy_scores.append(score_greedy)
        
        # 2. تقييم وكيل التعلم المعزز العميق (DQN)[cite: 1]
        obs_dqn, _ = env.reset(seed=seed)
        score_dqn = 0
        terminated = False
        while not terminated:
            action = dqn_agent.act(obs_dqn, evaluate=True) 
            obs_dqn, reward, terminated, _, _ = env.step(action)
            score_dqn += reward
        dqn_scores.append(score_dqn)
        
        # 3. تقييم البرمجة الديناميكية (DP Solver) للحصول على الحل الأمثل[cite: 1]
        score_dp = dp_solver.solve(tasks_this_episode)
        dp_scores.append(score_dp)
        
        print(f"Episode {episode+1:02d} | Greedy: {score_greedy:7.2f} | DQN: {score_dqn:7.2f} | DP (Optimal): {score_dp:7.2f}")

    avg_greedy = np.mean(greedy_scores)
    avg_dqn = np.mean(dqn_scores)
    avg_dp = np.mean(dp_scores)
    
    print("-" * 70)
    print(f"Average Greedy Reward: {avg_greedy:.2f}")
    print(f"Average DQN Reward:    {avg_dqn:.2f}")
    print(f"Average DP Reward:     {avg_dp:.2f} (Theoretical Maximum)")
    
    # إنشاء مجلد الرسوم البيانية إذا لم يكن موجوداً
    os.makedirs(plots_dir, exist_ok=True)
    
    # رسم المقارنة الثلاثية وحفظها للتقرير الأكاديمي[cite: 1]
    labels = ['Greedy', 'DQN', 'DP (Oracle)']
    averages = [avg_greedy, avg_dqn, avg_dp]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, averages, color=['#FF9999', '#66B2FF', '#99FF99'])
    plt.title('Performance Comparison: Greedy vs DQN vs DP')
    plt.ylabel('Average Total Reward')
    
    # إضافة الأرقام فوق الأعمدة
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, round(yval, 2), ha='center', va='bottom', fontweight='bold')
        
    plot_save_path = os.path.join(plots_dir, "comparison_bar.png")
    plt.savefig(plot_save_path)
    print(f"Comparison chart saved successfully at: {plot_save_path}")

if __name__ == "__main__":
    evaluate_models(num_episodes=20)