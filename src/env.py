import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ResourceAllocationEnv(gym.Env):
    """
    بيئة مخصصة لتخصيص الموارد تعتمد على مسألة حقيبة الظهر الديناميكية (Adaptive Knapsack).
    الهدف: تعظيم العائد من خلال قبول المهام المناسبة ضمن السعة المحددة للموارد (CPU و RAM).
    """
    
    def __init__(self, max_cpu=100.0, max_ram=100.0, max_steps=50):
        super(ResourceAllocationEnv, self).__init__()
        
        # إعدادات البيئة الأساسية (سعة الموارد وعدد المهام في كل حلقة التدريب)
        self.max_cpu = max_cpu
        self.max_ram = max_ram
        self.max_steps = max_steps
        
        # 1. فضاء الإجراءات (Action Space):
        # 0 = رفض المهمة (Reject)
        # 1 = قبول المهمة وتخصيص الموارد (Accept)
        self.action_space = spaces.Discrete(2)
        
        # 2. فضاء الحالة (Observation/State Space) مع التطبيع (Normalization):
        # المصفوفة تتكون من 5 قيم محصورة تقريباً بين 0.0 و 1.0 (أو أعلى قليلاً للعائد)
        # [نسبة_CPU_المتبقي, نسبة_RAM_المتبقي, نسبة_CPU_للمهمة, نسبة_RAM_للمهمة, نسبة_العائد]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """
        إعادة ضبط البيئة لبدء حلقة (Episode) جديدة.
        """
        super().reset(seed=seed, options=options)
        
        # استعادة السعة القصوى للموارد
        self.current_cpu = self.max_cpu
        self.current_ram = self.max_ram
        self.current_step = 0
        
        # توليد أول مهمة
        self._generate_task()
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def _generate_task(self):
        """
        توليد مهمة جديدة بمتطلبات وعوائد ديناميكية (عشوائية).
        """
        # متطلبات عشوائية بين 5 و 30 وحدة
        self.task_cpu = self.np_random.uniform(5.0, 30.0)
        self.task_ram = self.np_random.uniform(5.0, 30.0)
        
        # حساب العائد بناءً على المتطلبات مع إضافة معامل تشويش (Noise) 
        base_reward = (self.task_cpu + self.task_ram)
        noise = self.np_random.uniform(0.5, 1.5) 
        self.task_reward = base_reward * noise

    def _get_obs(self):
        """
        إرجاع الحالة الحالية للبيئة كـ Numpy Array مع تطبيع البيانات (Normalization)
        لتحسين وتسريع تعلم الشبكة العصبية.
        """
        return np.array([
            self.current_cpu / self.max_cpu,          # نسبة الـ CPU المتبقية
            self.current_ram / self.max_ram,          # نسبة الـ RAM المتبقية
            self.task_cpu / self.max_cpu,             # حجم مهمة CPU كنسبة
            self.task_ram / self.max_ram,             # حجم مهمة RAM كنسبة
            self.task_reward / 100.0                  # تصغير حجم العائد 
        ], dtype=np.float32)

    def step(self, action):
        """
        تنفيذ الإجراء المختار من قبل الوكيل وتحديث البيئة.
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # معالجة الإجراء
        if action == 1:  # الوكيل يقرر قبول المهمة
            # التحقق من توفر الموارد الكافية
            if self.task_cpu <= self.current_cpu and self.task_ram <= self.current_ram:
                # خصم الموارد وإعطاء المكافأة
                self.current_cpu -= self.task_cpu
                self.current_ram -= self.task_ram
                reward = float(self.task_reward)
            else:
                # عقوبة (Penalty)
                reward = -10.0 
        else: # الوكيل يقرر رفض المهمة
            reward = 0.0
            
        # التقدم للخطوة التالية
        self.current_step += 1
        
        # التحقق من انتهاء الحلقة
        if self.current_step >= self.max_steps:
            terminated = True
        else:
            self._generate_task()
            
        obs = self._get_obs()
        info = {
            "current_cpu": self.current_cpu,
            "current_ram": self.current_ram,
            "is_valid_action": reward > 0
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        طباعة حالة البيئة الحالية (مفيدة أثناء تصحيح الأخطاء Debugging).
        """
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Remaining Resources -> CPU: {self.current_cpu:.1f} | RAM: {self.current_ram:.1f}")
        print(f"Current Task -> CPU Req: {self.task_cpu:.1f} | RAM Req: {self.task_ram:.1f} | Reward: {self.task_reward:.1f}")
        print("-" * 40)