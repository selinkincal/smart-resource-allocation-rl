# compare.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from env import ResourceAllocationEnv
from greedy import GreedyAgent
from dqn_agent import DQNAgent
from dp_solver import DPSolver

def evaluate_models(num_episodes=20):
    # === ORTAMIN HAZIRLANMASI ===
    # Kaynak tahsisi ortamı oluşturuluyor (CPU ve RAM kapasiteleri 100 birim, maksimum 50 adım)
    # Bu ortam, Dinamik Çanta (Dynamic Knapsack) problemini simüle eder
    env = ResourceAllocationEnv(max_cpu=100.0, max_ram=100.0, max_steps=50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Kaydedilmiş modellerin ve grafiklerin bulunduğu/oluşturulacağı klasör yolları
    models_dir = os.path.join(project_root, "saved_models")
    plots_dir = os.path.join(project_root, "plots")
    
    # === AGENT'LARIN (AKTÖRLERİN) HAZIRLANMASI ===
    # 1. Açgözlü (Greedy) Agent: Her adımda anlık en iyi kararı verir, eğitim gerektirmez
    greedy_agent = GreedyAgent(threshold=1.0)
    
    # 2. DQN Agent: Derin Q-Öğrenmesi ile eğitilmiş sinir ağı tabanlı agent
    dqn_agent = DQNAgent(state_size=5, action_size=2)
    model_path = os.path.join(models_dir, "dqn_model.pth")
    
    # Daha önce eğitilmiş model var mı kontrol et
    if os.path.exists(model_path):
        # Varsa, kaydedilmiş ağırlıkları yükle
        dqn_agent.qnetwork_local.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded trained model from: {model_path}")
    else:
        print("Warning: Trained model not found. DQN will act with random weights!")
        
    # Değerlendirme moduna al 
    dqn_agent.qnetwork_local.eval() 
    
    # 3. DP (Dinamik Programlama) Çözücü: Teorik optimum çözümü hesaplar (Oracle/Referans)
    dp_solver = DPSolver(max_cpu=100.0, max_ram=100.0)
    
    # Her bir algoritmanın skorlarını toplayacağımız listeler
    greedy_scores = []
    dqn_scores = []
    dp_scores = []
    
    print("Starting the comprehensive final comparison (Greedy vs DQN vs DP)...")
    print("-" * 70)
    
    # Belirtilen sayıda bölüm (episode) boyunca test yap
    for episode in range(num_episodes):
        # Her bölüm için farklı ama tekrarlanabilir rastgelelik (seed) kullan
        seed = 42 + episode
        
        # === 1. GREEDY (AÇGÖZLÜ) ALGORİTMASININ DEĞERLENDİRİLMESİ ===
        obs_greedy, _ = env.reset(seed=seed)  # Ortamı sıfırla, ilk gözlemi al
        score_greedy = 0
        terminated = False
        tasks_this_episode = []  # Bu bölümdeki tüm görevleri sakla (DP için)
        
        while not terminated:
            # Gelen görevi kaydet: DP daha sonra tüm görevleri görüp optimal seçimi yapacak
            tasks_this_episode.append((env.task_cpu, env.task_ram, env.task_reward))
            
            # Açgözlü agent'tan aksiyon al (0: reddet, 1: kabul et)
            action = greedy_agent.select_action(obs_greedy)
            # Aksiyonu ortamda uygula
            obs_greedy, reward, terminated, _, _ = env.step(action)
            score_greedy += reward
        greedy_scores.append(score_greedy)
        
        # === 2. DQN (DERİN Q-ÖĞRENMESİ) AGENT'ININ DEĞERLENDİRİLMESİ ===
        obs_dqn, _ = env.reset(seed=seed)  # Aynı seed ile aynı görev sırası garanti edilir
        score_dqn = 0
        terminated = False
        while not terminated:
            # evaluate=True: epsilon-greedy yapma, tamamen deterministik karar ver
            action = dqn_agent.act(obs_dqn, evaluate=True) 
            obs_dqn, reward, terminated, _, _ = env.step(action)
            score_dqn += reward
        dqn_scores.append(score_dqn)
        
        # === 3. DP (DİNAMİK PROGRAMLAMA) İLE OPTİMAL ÇÖZÜM ===
        # Bu bölümde görülen tüm görevleri ver, DP en iyi kombinasyonu hesaplasın
        score_dp = dp_solver.solve(tasks_this_episode)
        dp_scores.append(score_dp)
        
        # Her bölümün sonuçlarını ekrana yaz
        print(f"Episode {episode+1:02d} | Greedy: {score_greedy:7.2f} | DQN: {score_dqn:7.2f} | DP (Optimal): {score_dp:7.2f}")

    # === ORTALAMALARIN HESAPLANMASI ===
    avg_greedy = np.mean(greedy_scores)
    avg_dqn = np.mean(dqn_scores)
    avg_dp = np.mean(dp_scores)
    
    print("-" * 70)
    print(f"Average Greedy Reward: {avg_greedy:.2f}")
    print(f"Average DQN Reward:    {avg_dqn:.2f}")
    print(f"Average DP Reward:     {avg_dp:.2f} (Theoretical Maximum)")
    
    # === GRAFİK OLUŞTURMA VE KAYDETME ===
    # Grafiklerin kaydedileceği klasör yoksa oluştur
    os.makedirs(plots_dir, exist_ok=True)
    
    # Çubuk grafik (bar chart) ile üç algoritmanın performans karşılaştırması
    labels = ['Greedy', 'DQN', 'DP (Oracle)']
    averages = [avg_greedy, avg_dqn, avg_dp]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, averages, color=['#FF9999', '#66B2FF', '#99FF99'])
    plt.title('Performance Comparison: Greedy vs DQN vs DP')
    plt.ylabel('Average Total Reward')
    
    # Çubukların üzerine sayısal değerleri yaz
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, round(yval, 2), ha='center', va='bottom', fontweight='bold')
        
    plot_save_path = os.path.join(plots_dir, "comparison_bar.png")
    plt.savefig(plot_save_path)
    print(f"Comparison chart saved successfully at: {plot_save_path}")

if __name__ == "__main__":
    evaluate_models(num_episodes=20)