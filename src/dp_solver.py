import numpy as np

class DPSolver:
    """
    مُحلل يعتمد على البرمجة الديناميكية (Dynamic Programming).
    يحل مسألة حقيبة الظهر ثنائية الأبعاد (2D Knapsack) حيث القيود هي CPU و RAM.
    يُستخدم لحساب "الحل الأمثل النظري" (Theoretical Optimal) للمقارنة.
    """
    def __init__(self, max_cpu=100.0, max_ram=100.0):
        # البرمجة الديناميكية تتطلب أرقاماً صحيحة لحجم المصفوفة
        # نقوم بتقريب السعة القصوى لتكون أعداداً صحيحة (Integers)
        self.max_cpu = int(round(max_cpu))
        self.max_ram = int(round(max_ram))

    def solve(self, tasks):
        """
        تستقبل قائمة من المهام وتُرجع أقصى عائد ممكن تحقيقه.
        tasks: قائمة من الـ Tuples بالشكل (task_cpu, task_ram, task_reward)
        """
        # تهيئة مصفوفة ثنائية الأبعاد بالأصفار
        # dp[c][r] ستحتفظ بأقصى عائد يمكن تحقيقه باستخدام c من الـ CPU و r من الـ RAM
        dp = np.zeros((self.max_cpu + 1, self.max_ram + 1), dtype=np.float32)

        for cpu_req, ram_req, reward in tasks:
            # تقريب متطلبات المهمة إلى أعداد صحيحة
            c_req = int(round(cpu_req))
            r_req = int(round(ram_req))
            
            # إذا كانت متطلبات المهمة أكبر من السعة القصوى للبيئة، نتجاهلها
            if c_req > self.max_cpu or r_req > self.max_ram:
                continue

            # المرور على المصفوفة بشكل عكسي (Backwards) لضمان عدم استخدام نفس المهمة أكثر من مرة
            # (وهذا يمثل مسألة 0/1 Knapsack)
            for c in range(self.max_cpu, c_req - 1, -1):
                for r in range(self.max_ram, r_req - 1, -1):
                    dp[c][r] = max(dp[c][r], dp[c - c_req][r - r_req] + reward)

        # الحل الأمثل سيكون في الزاوية الأخيرة من المصفوفة
        optimal_reward = dp[self.max_cpu][self.max_ram]
        return optimal_reward

# قسم اختباري للتحقق من سلامة الخوارزمية
if __name__ == "__main__":
    # توليد قائمة مهام وهمية لاختبار الخوارزمية
    sample_tasks = [
        (20.5, 30.1, 100.0), # مهمة 1
        (50.0, 40.0, 250.0), # مهمة 2
        (40.0, 50.0, 200.0), # مهمة 3
        (10.0, 10.0, 80.0)   # مهمة 4
    ]
    
    print("Starting Dynamic Programming Solver test...")
    solver = DPSolver(max_cpu=100.0, max_ram=100.0)
    optimal_score = solver.solve(sample_tasks)
    
    print("-" * 40)
    print(f"Number of tasks evaluated: {len(sample_tasks)}")
    print(f"Theoretical Optimal Reward: {optimal_score:.2f}")