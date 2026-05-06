import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. تعريف المعمارية الخاصة بالشبكة العصبية (Q-Network)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        # طبقات الشبكة العصبية (ANN)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. ذاكرة التجربة (Replay Buffer)
class ReplayBuffer:
    """
    تُستخدم لتخزين تجارب الوكيل السابقة (حالة، إجراء، مكافأة، حالة تالية)
    لكي يتعلم النموذج من عينات عشوائية ويتجنب نسيان ما تعلمه مسبقاً.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# 3. الوكيل الذكي (DQN Agent)
class DQNAgent:
    def __init__(self, state_size=5, action_size=2, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        
        # إعدادات التعلم (Hyperparameters)
        self.gamma = 0.99           # عامل الخصم (Discount factor) للمكافآت المستقبلية
        self.lr = 1e-3              # معدل التعلم (Learning rate)
        self.batch_size = 64        # حجم العينة المسحوبة من الذاكرة في كل خطوة تدريب
        self.update_every = 4       # تحديث الشبكة كل 4 خطوات
        
        # استكشاف vs استغلال (Epsilon-greedy)
        self.epsilon = 1.0          # قيمة الاستكشاف المبدئية (100% عشوائي)
        self.epsilon_min = 0.01     # الحد الأدنى للاستكشاف
        self.epsilon_decay = 0.995  # معدل تناقص الاستكشاف

        # تهيئة الشبكات: 
        # نستخدم شبكتين، واحدة للقرارات الحالية (Local) وأخرى لحساب الهدف (Target) لزيادة استقرار التدريب
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(capacity=100000)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # حفظ التجربة في الذاكرة
        self.memory.push(state, action, reward, next_state, done)

        # التعلم كل عدد محدد من الخطوات (update_every)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma)

    def act(self, state, evaluate=False):
        """
        اتخاذ القرار بناءً على الحالة الحالية.
        إذا كنا في مرحلة التقييم (evaluate=True)، فلن يتم اتخاذ أي قرارات عشوائية.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval() # تفعيل وضع التقييم
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # العودة لوضع التدريب

        # Epsilon-greedy action selection
        if not evaluate and random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        elif evaluate:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        تحديث أوزان الشبكة (Q-Network) باستخدام عينة من التجارب.
        """
        states, actions, rewards, next_states, dones = experiences

        # تحويل البيانات إلى Tensors لتتوافق مع PyTorch
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long().unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float().unsqueeze(1)

        # 1. الحصول على أقصى قيمة متوقعة (Q-value) للحالة التالية من الشبكة الهدف (Target Network)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 2. حساب قيمة الهدف الحالية (Q Target)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 3. الحصول على التوقعات الحالية من الشبكة المحلية (Local Network)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 4. حساب مقدار الخطأ (Loss) وتحديث الأوزان (Backpropagation)
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. تحديث الشبكة الهدف بشكل ناعم (Soft Update)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        """
        تحديث تدريجي لأوزان الشبكة الهدف لتجنب التقلبات الحادة.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        """
        تقليل نسبة الاستكشاف تدريجياً بمرور الوقت.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)