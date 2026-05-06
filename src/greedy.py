import numpy as np

class GreedyAgent:
    """
    وكيل يعتمد على الخوارزمية الجشعة (Greedy Algorithm).
    يتخذ القرار بناءً على توفر الموارد ونسبة العائد إلى التكلفة في اللحظة الحالية.
    """
    def __init__(self, threshold=1.0):
        # الحد الأدنى لنسبة العائد إلى الموارد لقبول المهمة.
        # القيمة 1.0 تعني أن العائد يجب أن يكون مساوياً أو أكبر من مجموع الموارد المستهلكة.
        self.threshold = threshold

    def select_action(self, obs):
        """
        استقبال ملاحظة البيئة الحالية (Observation) واتخاذ قرار (0 للرفض، 1 للقبول).
        """
        current_cpu, current_ram, task_cpu, task_ram, task_reward = obs
        
        # 1. التحقق من توفر الموارد الكافية
        if task_cpu <= current_cpu and task_ram <= current_ram:
            
            # 2. حساب كفاءة المهمة (نسبة العائد إلى التكلفة)
            cost = task_cpu + task_ram
            efficiency = task_reward / cost if cost > 0 else 0
            
            # 3. اتخاذ القرار الجشع (Greedy Decision)
            if efficiency >= self.threshold:
                return 1 # قبول المهمة
                
        return 0 # رفض المهمة

    def evaluate(self, env, num_episodes=10):
        """
        اختبار الوكيل الجشع على البيئة لعدد محدد من الحلقات (Episodes)
        وإرجاع متوسط العائد، ليتم استخدامه كمعيار مقارنة (Baseline).
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

# قسم اختباري لتشغيل البيئة والوكيل معاً والتحقق من سلامة الكود
if __name__ == "__main__":
    from env import ResourceAllocationEnv
    
    # تهيئة البيئة
    env = ResourceAllocationEnv(max_cpu=100.0, max_ram=100.0, max_steps=50)
    
    # تهيئة الوكيل الجشع
    greedy_agent = GreedyAgent(threshold=1.0)
    
    # تقييم أداء الوكيل الجشع
    print("Starting Greedy Baseline evaluation...")
    greedy_agent.evaluate(env, num_episodes=5)