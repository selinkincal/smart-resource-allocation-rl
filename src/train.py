# train.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from env import ResourceAllocationEnv
from dqn_agent import DQNAgent

def train(num_episodes=3000):
    """
    DQN ajanını eğitir.
    num_episodes: Kaç bölüm boyunca eğitileceği
    """
    # 1. ORTAM VE AJAN HAZIRLIĞI
    # Kaynak tahsisi ortamı: 100 CPU, 100 RAM, her bölümde 50 görev
    env = ResourceAllocationEnv(max_cpu=100.0, max_ram=100.0, max_steps=50)
    agent = DQNAgent(state_size=5, action_size=2)
    
    scores = []  # Her bölümün toplam ödülünü sakla
    
    print("Starting DQN agent training...")
    print("-" * 40)
    
    # EĞİTİM DÖNGÜSÜ
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()  # Yeni bölüm başlat
        score = 0
        terminated = False
        
        # Bölüm bitene kadar (50 adım) devam et
        while not terminated:
            # Ajan, mevcut duruma göre aksiyon seç (epsilon-greedy ile)
            action = agent.act(obs)
            
            # Aksiyonu ortamda uygula
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Ajan bu deneyimden öğren (deneyimi hafızaya kaydet + zamanı gelirse ağı güncelle)
            agent.step(obs, action, reward, next_obs, terminated)
            
            # Bir sonraki adıma geç
            obs = next_obs
            score += reward
            
        # Bölüm sonu işlemleri
        agent.update_epsilon()  # Keşif oranını azalt (zamanla daha az rastgele hareket)
        scores.append(score)
        
        # Her 50 bölümde bir ilerleme raporu yazdır
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])  # Son 50 bölümün ortalaması
            print(f"Episode {episode}/{num_episodes} | Average Score (last 50): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("-" * 40)
    print("Training completed successfully!")
    
    return scores, agent

def save_results(scores, agent):
    """
    Eğitilmiş modeli ve öğrenme eğrisi grafiğini kaydeder.
    """
    # === DOSYA YOLU AYARLARI ===
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Bir üst dizin (proje kökü)
    
    models_dir = os.path.join(project_root, "saved_models")
    plots_dir = os.path.join(project_root, "plots")
    
    # Klasör yoksa oluştur
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # === MODELİ KAYDET ===
    model_path = os.path.join(models_dir, "dqn_model.pth")
    torch.save(agent.qnetwork_local.state_dict(), model_path)
    print(f"Trained model saved at: {model_path}")
    
    # === ÖĞRENME EĞRİSİ (LEARNING CURVE) GRAFİĞİ ===
    plt.figure(figsize=(10, 6))
    # Ham skorlar (yarı saydam mavi)
    plt.plot(np.arange(len(scores)), scores, color='blue', alpha=0.6)
    
    # Hareketli ortalama (Moving Average) - eğilimi görmek için
    window = 20
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    # moving_avg dizisi window-1 kadar kısadır, bu yüzden x eksenini ayarla
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
    # 5000 bölüm boyunca eğit (önceki koddan farklı: train.py'de num_episodes=5000)
    training_scores, trained_agent = train(num_episodes=5000)
    
    # Sonuçları kaydet
    save_results(training_scores, trained_agent)