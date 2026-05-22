# greedy.py
import numpy as np

class GreedyAgent:
    """
    Açgözlü (Greedy) algoritma tabanlı ajan.
    Her adımda anlık olarak en iyi görünen kararı verir.
    Hiçbir eğitim yapmaz, sadece elindeki kurala göre hareket eder.
    Basit bir referans (baseline) olarak kullanılır.
    """
    def __init__(self, threshold=1.0):
        # threshold: Ödül/maliyet oranı bu değerin üstündeyse görevi kabul et
        # threshold=1.0: Ödül, harcanan kaynaklardan (CPU+RAM) fazla veya eşitse kabul et
        self.threshold = threshold

    def select_action(self, obs):
        """
        Gözleme (observation) göre aksiyon seç.
        obs: [kalan_CPU, kalan_RAM, görev_CPU, görev_RAM, görev_ödül]
        
        Mantık:
        1. Kaynak yeterli değilse -> REDDET
        2. Kaynak yeterliyse, verimlilik (ödül/maliyet) hesapla
        3. Verimlilik threshold değerinden büyük veya eşitse -> KABUL ET, yoksa REDDET
        """
        # Gözlem vektörünü bileşenlerine ayır
        current_cpu, current_ram, task_cpu, task_ram, task_reward = obs
        
        # 1. Kaynak yeterlilik kontrolü (en temel kısıt)
        if task_cpu <= current_cpu and task_ram <= current_ram:
            
            # 2. Verimlilik (Efficiency) = Ödül / Toplam Kaynak Tüketimi
            cost = task_cpu + task_ram
            efficiency = task_reward / cost if cost > 0 else 0
            
            # 3. Açgözlü karar: eşik değerini geçiyorsa kabul et
            if efficiency >= self.threshold:
                return 1  # KABUL ET
                
        return 0  # REDDET

    def evaluate(self, env, num_episodes=10):
        """
        Açgözlü ajanı belirli sayıda bölümde test et.
        Ortalama ödülü hesapla. Bu, DQN gibi gelişmiş yöntemlerle karşılaştırma için baz oluşturur.
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            terminated = False
            
            while not terminated:
                action = self.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")
            
        avg_reward = np.mean(total_rewards)
        print("-" * 40)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward

# Basit bir test: Açgözlü ajanı ortamda çalıştır
if __name__ == "__main__":
    from env import ResourceAllocationEnv
    
    env = ResourceAllocationEnv(max_cpu=100.0, max_ram=100.0, max_steps=50)
    greedy_agent = GreedyAgent(threshold=1.0)
    
    print("Starting Greedy Baseline evaluation...")
    greedy_agent.evaluate(env, num_episodes=5)