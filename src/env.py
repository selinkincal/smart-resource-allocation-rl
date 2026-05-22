# env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ResourceAllocationEnv(gym.Env):
    """
    Özel kaynak tahsisi ortamı: Dinamik Çanta Problemini (Dynamic Knapsack) simüle eder.
    Hedef: Sınırlı CPU ve RAM kaynaklarıyla, gelen görevler arasından en yüksek ödülü toplamak.
    Her görev ya KABUL edilir (kaynak tüketir, ödül verir) ya da REDdedilir (kaynak tüketmez).
    """
    
    def __init__(self, max_cpu=100.0, max_ram=100.0, max_steps=50):
        super(ResourceAllocationEnv, self).__init__()
        
        # Ortam sabitleri
        self.max_cpu = max_cpu      # Maksimum CPU kapasitesi (birim)
        self.max_ram = max_ram      # Maksimum RAM kapasitesi (birim)
        self.max_steps = max_steps  # Bir bölümdeki maksimum görev sayısı
        
        # 1. AKSİYON UZAYI (Action Space)
        # 0 = GÖREVİ REDDET
        # 1 = GÖREVİ KABUL ET (kaynak yeterliyse)
        self.action_space = spaces.Discrete(2)
        
        # 2. GÖZLEM UZAYI (Observation/State Space)
        # 5 boyutlu vektör: [kalan_CPU_oranı, kalan_RAM_oranı, görev_CPU_oranı, görev_RAM_oranı, görev_ödül_normalize]
        # Normalizasyon (0-1 arası) sinir ağlarının daha hızlı ve kararlı öğrenmesi için yapılır
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32),  # Ödül 10'a kadar normalize edilmiş
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """
        Ortamı başlangıç durumuna sıfırla.
        Yeni bir bölüm (episode) başlatmak için çağrılır.
        """
        super().reset(seed=seed, options=options)  # Rastgele sayı üreteci seed'ini ayarla
        
        # Kaynakları tamamen doldur
        self.current_cpu = self.max_cpu
        self.current_ram = self.max_ram
        self.current_step = 0
        
        # İlk görevi oluştur
        self._generate_task()
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def _generate_task(self):
        """
        Yeni bir görev oluşturur.
        CPU ve RAM talepleri 5-25 birim arasında rastgele.
        Ödül = (CPU_talep + RAM_talep) * gürültü (0.5-1.5 arası)
        """
        # Rastgele talepler (5.0 ile 25.0 arası)
        self.task_cpu = self.np_random.uniform(5.0, 25.0)
        self.task_ram = self.np_random.uniform(5.0, 25.0)
        
        # Ödül hesaplama: daha büyük görevler genelde daha çok ödül verir, ama rastgele gürültü var
        base_reward = (self.task_cpu + self.task_ram)
        noise = self.np_random.uniform(0.5, 1.5) 
        self.task_reward = base_reward * noise

    def _get_obs(self):
        """
        Mevcut durumu normalize edilmiş vektör olarak döndür.
        Normalizasyon: Q-network'ün daha iyi öğrenmesini sağlar.
        """
        return np.array([
            self.current_cpu / self.max_cpu,          # Kalan CPU oranı (0-1)
            self.current_ram / self.max_ram,          # Kalan RAM oranı (0-1)
            self.task_cpu / self.max_cpu,             # Görevin CPU talebi (oran)
            self.task_ram / self.max_ram,             # Görevin RAM talebi (oran)
            self.task_reward / 50.0                   # Normalize edilmiş ödül (max yaklaşık 50 olabilir)
        ], dtype=np.float32)

    def step(self, action):
        """
        Verilen aksiyonu ortamda uygula.
        Aksiyon: 0=Reddet, 1=Kabul et
        
        Dönüş değerleri:
        - obs: Yeni gözlem
        - reward: Bu adımda kazanılan ödül
        - terminated: Bölüm bitti mi? (max_steps'e ulaşıldı mı?)
        - truncated: Zaman aşımı (bu ortamda kullanılmıyor)
        - info: Ek bilgi (hata ayıklama için)
        """
        reward = 0.0
        terminated = False
        truncated = False  # Bu ortamda zaman aşımı yok, sadece terminated var
        
        # AKSİYONU İŞLE
        if action == 1:  # KABUL ET
            # Kaynak yeterli mi kontrol et
            if self.task_cpu <= self.current_cpu and self.task_ram <= self.current_ram:
                # Kaynakları düş, ödülü ver
                self.current_cpu -= self.task_cpu
                self.current_ram -= self.task_ram
                reward = float(self.task_reward)
            else:
                # Kaynak yetersiz: CEZA (-5)
                # Bu, agent'a geçersiz aksiyonun kötü olduğunu öğretir
                reward = -5.0 
        else:  # REDDET (action == 0)
            # Reddetmenin hiçbir maliyeti yok, ödül de yok
            reward = 0.0
            
        # ZAMAN ADIMINI İLERLET
        self.current_step += 1
        
        # BÖLÜM SONMU KONTROLÜ
        if self.current_step >= self.max_steps:
            terminated = True  # Maksimum adıma ulaşıldı, bölüm biter
        else:
            # Yeni görev oluştur (bir sonraki adım için)
            self._generate_task()
            
        # Yeni gözlemi hesapla
        obs = self._get_obs()
        
        # Hata ayıklama için ek bilgi
        info = {
            "current_cpu": self.current_cpu,
            "current_ram": self.current_ram,
            "is_valid_action": reward > 0  # Kabul edildi ve kaynak yeterliydiyse True
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Ortamın mevcut durumunu insan okunabilir formatta yazdırır.
        Hata ayıklama ve geliştirme sırasında kullanılır.
        """
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Remaining Resources -> CPU: {self.current_cpu:.1f} | RAM: {self.current_ram:.1f}")
        print(f"Current Task -> CPU Req: {self.task_cpu:.1f} | RAM Req: {self.task_ram:.1f} | Reward: {self.task_reward:.1f}")
        print("-" * 40)