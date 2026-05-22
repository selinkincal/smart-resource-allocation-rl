import numpy as np

class DPSolver:
    """
    Dinamik Programlama tabanlı çözümleyici.
    2B Sırt Çantası (Knapsack) problemini CPU ve RAM kısıtları ile çözer.
    """
    def __init__(self, max_cpu=100.0, max_ram=100.0):
        # DP algoritması tam sayı dizinleri gerektirdiği için kapasiteleri yuvarlıyoruz
        self.max_cpu = int(round(max_cpu))
        self.max_ram = int(round(max_ram))

    def solve(self, tasks):
        """
        tasks: (cpu, ram, reward) formatında liste.
        """
        # DP tablosu: [cpu][ram] koordinatlarında elde edilebilecek maksimum ödülü saklar
        dp = np.zeros((self.max_cpu + 1, self.max_ram + 1), dtype=np.float32)

        for cpu_req, ram_req, reward in tasks:
            c_req = int(round(cpu_req))
            r_req = int(round(ram_req))
            
            # Kapasiteyi aşan görevlerin elenmesi
            if c_req > self.max_cpu or r_req > self.max_ram:
                continue

            # 0/1 Sırt Çantası mantığı: Her nesne sadece bir kez kullanılabilir, bu yüzden ters döngü
            for c in range(self.max_cpu, c_req - 1, -1):
                for r in range(self.max_ram, r_req - 1, -1):
                    dp[c][r] = max(dp[c][r], dp[c - c_req][r - r_req] + reward)

        # Matrisin en alt sağ köşesi teorik en yüksek getiriyi barındırır
        optimal_reward = dp[self.max_cpu][self.max_ram]
        return optimal_reward

if __name__ == "__main__":
    sample_tasks = [
        (20.5, 30.1, 100.0),
        (50.0, 40.0, 250.0),
        (40.0, 50.0, 200.0),
        (10.0, 10.0, 80.0)
    ]
    
    print("Starting Dynamic Programming Solver test...")
    solver = DPSolver(max_cpu=100.0, max_ram=100.0)
    optimal_score = solver.solve(sample_tasks)
    
    print("-" * 40)
    print(f"Number of tasks evaluated: {len(sample_tasks)}")
    print(f"Theoretical Optimal Reward: {optimal_score:.2f}")