import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from env import ResourceAllocationEnv
from dqn_agent import DQNAgent

def train(num_episodes=500):
    # 1. تهيئة البيئة والوكيل بناءً على متطلبات مشروع حقيبة الظهر
    env = ResourceAllocationEnv(max_cpu=100.0, max_ram=100.0, max_steps=50)
    agent = DQNAgent(state_size=5, action_size=2)
    
    scores = [] # لتخزين مجموع العوائد في كل حلقة
    
    print("Starting DQN agent training...")
    print("-" * 40)
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        score = 0
        terminated = False
        
        while not terminated:
            # الوكيل يختار إجراء بناءً على الحالة الحالية
            action = agent.act(obs)
            
            # تنفيذ الإجراء في البيئة ومراقبة النتيجة
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # الوكيل يتعلم من هذه التجربة (Deep Reinforcement Learning)[cite: 1]
            agent.step(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            score += reward
            
        # تحديث نسبة الاستكشاف (Epsilon) في نهاية كل حلقة
        agent.update_epsilon()
        scores.append(score)
        
        # طباعة ملخص كل 50 حلقة
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}/{num_episodes} | Average Score (last 50): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("-" * 40)
    print("Training completed successfully!")
    
    return scores, agent

def save_results(scores, agent):
    # الحصول على المسار الصحيح للمجلد الرئيسي للمشروع
    # أينما كان ملف train.py، سنعود خطوة للخلف لإنشاء المجلدات في المجلد الرئيسي
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    models_dir = os.path.join(project_root, "saved_models")
    plots_dir = os.path.join(project_root, "plots")
    
    # إنشاء المجلدات إذا لم تكن موجودة بشكل آمن
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # حفظ النموذج (Weights) المدرب
    model_path = os.path.join(models_dir, "dqn_model.pth")
    torch.save(agent.qnetwork_local.state_dict(), model_path)
    print(f"Trained model saved at: {model_path}")
    
    # رسم منحنى التعلم وحفظه للتقرير الأكاديمي[cite: 1]
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, color='blue', alpha=0.6)
    
    # رسم متوسط متحرك (Moving Average) لتوضيح منحنى التحسن
    window = 20
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(scores)), moving_avg, color='red', linewidth=2, label='Moving Average (20 episodes)')
    
    plt.title('DQN Learning Curve (Smart Resource Allocation)')
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(plots_dir, "reward_curve.png")
    plt.savefig(plot_path)
    print(f"Learning curve plot saved at: {plot_path}")

if __name__ == "__main__":
    # تشغيل التدريب وفقاً لسيناريو "Learning to Optimize" المطلوب[cite: 1]
    training_scores, trained_agent = train(num_episodes=500)
    
    # حفظ النتائج النهائية
    save_results(training_scores, trained_agent)