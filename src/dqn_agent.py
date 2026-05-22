# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ============================================
# 1. Q-NETWORK (DERİN SİNİR AĞI)
# ============================================
class QNetwork(nn.Module):
    """Durum (state) alır, her aksiyon için Q-değeri (beklenen toplam ödül) döndürür."""
    def __init__(self, state_size, action_size, hidden_size=128, hidden_size2=64):
        super(QNetwork, self).__init__()
        # Giriş katmanı: state_size -> hidden_size (128 nöron)
        self.fc1 = nn.Linear(state_size, hidden_size)
        # Ara gizli katman: hidden_size -> hidden_size2 (64 nöron)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        # Çıkış katmanı: hidden_size2 -> action_size (Q değerleri)
        self.fc3 = nn.Linear(hidden_size2, action_size)

    def forward(self, state):
        """İleri yayılım (forward pass): ReLU aktivasyonu ile"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # lineer çıkış, Q-değerleri


# ============================================
# 2. DENEYİM HAVUZU (REPLAY BUFFER)
# ============================================
class ReplayBuffer:
    """
    Geçmiş deneyimleri (state, action, reward, next_state, done) saklar.
    Rastgele örnekleme yaparak deneyimleri ilişkisiz hale getirir.
    Bu, DQN'yi kararlı kılan kritik bir mekanizmadır.
    """
    def __init__(self, capacity=10000):
        # capacity: maksimum kaç deneyim saklanacağı
        # Deque: eski deneyimler otomatik silinir
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Yeni bir deneyim ekle"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Rastgele batch_size kadar deneyim seç"""
        batch = random.sample(self.buffer, batch_size)
        # Her bileşeni ayrı ayrı numpy dizilerine dönüştür
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


# ============================================
# 3. DQN AGENT (AKTÖR)
# ============================================
class DQNAgent:
    def __init__(self, state_size=5, action_size=2, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        
        # === HİPERPARAMETRELER ===
        self.gamma = 0.99         # Gelecek ödüllerin iskonto faktörü (0.99: geleceğe çok bakar)
        self.lr = 1e-3            # Öğrenme hızı (Learning rate)
        self.batch_size = 128     # Her güncellemede kaç deneyim kullanılacağı
        self.update_every = 4     # Hedef ağ (target network) ne sıklıkta güncellenecek
        
        # === EPSILON-GREEDY KEŞİF ===
        self.epsilon = 1.0        # Başlangıçta %100 rastgele hareket (keşif)
        self.epsilon_min = 0.01   # Minimum keşif oranı (%1)
        self.epsilon_decay = 0.9995  # Her bölümde epsilon azalma faktörü

        # === AĞLAR ===
        # Local network: aksiyonları seçmek ve güncellenmek için
        self.qnetwork_local = QNetwork(state_size, action_size)
        # Target network: sabit hedef Q-değerleri hesaplamak için (kararlılık)
        self.qnetwork_target = QNetwork(state_size, action_size)
        # Optimizer: Adam (gradient descent varyantı)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Deneyim havuzu
        self.memory = ReplayBuffer(capacity=100000)
        self.t_step = 0  # Kaç adım geçtiği (target network güncelleme sayacı)

    def step(self, state, action, reward, next_state, done):
        """Her aksiyon sonrası çağrılır. Deneyimi kaydeder ve zamanı gelince öğrenir."""
        # Deneyimi hafızaya ekle
        self.memory.push(state, action, reward, next_state, done)

        # Her update_every adımda bir öğren (target güncelleme)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Yeterli deneyim birikmişse öğren
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)

    def act(self, state, evaluate=False):
        """
        Duruma göre aksiyon seç.
        evaluate=True: sadece exploitation (keşif yok, epsilon kullanılmaz)
        evaluate=False ve random sayı > epsilon: exploitation (ağın tahmini)
        evaluate=False ve random sayı <= epsilon: exploration (rastgele aksiyon)
        """
        state = torch.from_numpy(state).float().unsqueeze(0)  # (1, state_size) boyutunda tensor
        self.qnetwork_local.eval()  # Değerlendirme modu (eğer dropout varsa devre dışı)
        with torch.no_grad():  # Gradyan hesaplama kapalı (daha hızlı)
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # Tekrar eğitim moduna al

        # Epsilon-greedy seçimi
        if not evaluate and random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())  # En yüksek Q-değerine sahip aksiyon
        elif evaluate:
            return np.argmax(action_values.cpu().data.numpy())  # Değerlendirmede hep en iyiyi seç
        else:
            return random.choice(np.arange(self.action_size))  # Rastgele aksiyon (keşif)

    def learn(self, experiences, gamma):
        """
        Deneyimleri kullanarak Q-network'ü güncelle.
        Burada Bellman denklemi ve Mean Squared Error (MSE) loss kullanılır.
        """
        states, actions, rewards, next_states, dones = experiences

        # Verileri PyTorch tensor'lerine dönüştür
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long().unsqueeze(1)  # (batch, 1) boyutunda
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float().unsqueeze(1)

        # 1. HEDEF Q-DEĞERİNİ HESAPLA
        # Q_target = reward + gamma * max_a' Q_target(next_state, a')
        # Target network'ten sonraki durumdaki maksimum Q-değeri alınır
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Eğer done=True ise (bölüm bitmiş), gelecek ödül yoktur -> (1 - dones) ile sıfırlanır
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 2. MEVCUT Q-DEĞERİNİ HESAPLA (yapılan aksiyona göre)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 3. KAYIP FONKSİYONU (LOSS) = MSE(Q_expected, Q_targets)
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()  # Önceki gradyanları sıfırla
        loss.backward()             # Geri yayılım (backpropagation)
        self.optimizer.step()       # Ağırlıkları güncelle

        # 4. HEDEF AĞI YUMUŞAK GÜNCELLE (SOFT UPDATE)
        # Target network'ü local network'e tamamen değil, küçük adımlarla yaklaştır
        # Bu, öğrenmeyi kararlı hale getirir
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        """
        target_param = tau * local_param + (1 - tau) * target_param
        tau küçük olduğunda (örn: 0.001) güncelleme çok yumuşak olur
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """Epsilon değerini azalt (keşif zamanla azalır, sömürü artar)"""
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)