# 🧠 Akıllı Kaynak Tahsisi (Smart Resource Allocation)

Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning) kullanılarak 2 Boyutlu Sırt Çantası (2D-Knapsack) problemi üzerinden dinamik kaynak optimizasyonu sağlayan yapay zeka projesi.

## 👥 Geliştirici Ekibi

| İsim | Rol |
|------|-----|
| **Louai ALDALATİ** | 💻 **Yazılım Geliştirme:** DQN, DP ve Greedy algoritmalarının Python ile kodlanması, model eğitimi ve ana mimarinin kurulması, kodların test edilmesi ve deneysel sonuçların analizi. |
| **Selin KINCAL** | 📄 **Akademik Raporlama:** Simülasyon ortamının (Environment) çalıştırılması, LaTeX ile proje bildirisinin (paper) IEEE formatında oluşturulması, kaynakça yönetimi ve dokümantasyon. |
| **Naciye KAYA** | ⚙️ **Test & Analiz:** Simülasyon ortamının (Environment) çalıştırılması, kodların test edilmesi ve deneysel sonuçların analizi. |

*Bu proje, Algoritma Analizi ve Tasarımı dersi kapsamında 3 kişilik ekip tarafından geliştirilmiştir.*

## 💻 Proje Hakkında

Bu çalışma, bulut bilişim gibi sınırlı kapasiteye sahip ortamlarda maksimum verimi elde etmek için gelen görevlerin (CPU ve RAM kısıtları altında) kabul veya ret kararlarının optimize edilmesini hedefler. 

Proje kapsamında 3 farklı yaklaşım karşılaştırılmaktadır:
- 🤖 **Derin Pekiştirmeli Öğrenme (DQN):** Deneyimlerden öğrenerek geleceğe dönük en iyi kararları veren yapay zeka ajanı.
- 📐 **Dinamik Programlama (DP):** Problemin teorik olarak en mükemmel (optimum) çözümünü sunan referans modeli.
- ⚡ **Açgözlü (Greedy) Algoritma:** Anlık olarak en yüksek verimlilik (ödül/maliyet) oranına sahip görevleri seçen geleneksel yaklaşım.

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Amaç |
|-----------|------|
| **Python** | Temel programlama dili |
| **PyTorch** | Derin Q-Ağları (DQN) ve sinir ağı mimarisi tasarımı |
| **Gymnasium** | Özel pekiştirmeli öğrenme ortamı (Environment) oluşturma |
| **NumPy** | Matris hesaplamaları ve veri manipülasyonu |
| **Matplotlib** | Öğrenme eğrisi ve performans karşılaştırma grafikleri |

## 📂 Proje Yapısı
proje_klasoru/
├── src/
│   ├── env.py          # Kaynak tahsisi Gymnasium ortamı
│   ├── dqn_agent.py    # DQN sinir ağı ve Replay Buffer mimarisi
│   ├── greedy.py       # Açgözlü (Greedy) ajan implementasyonu
│   ├── dp_solver.py    # Dinamik Programlama (Oracle) çözücüsü
│   ├── train.py        # Modeli eğitme ve kaydetme betiği
│   └── compare.py      # 3 algoritmayı karşılaştıran test betiği
├── saved_models/       # Eğitilmiş model ağırlıkları (.pth)
├── plots/              # Çıktı grafikleri (reward_curve.png vb.)
├── report/             # LaTeX proje raporu (main.tex, .bib)
└── README.md           # Proje dokümantasyonu


## ✨ Özellikler

### 🎯 Özelleştirilmiş Eğitim Ortamı (Environment)
- 📊 Dinamik CPU ve RAM kapasite takibi
- 🎲 Rastgelelik katsayısı ile görev gereksinimleri ve ödül simülasyonu
- ⚖️ Geçersiz/Kapasite aşan hamlelerde negatif ödül (ceza) mekanizması

### 🧠 Derin Q-Öğrenmesi (DQN) Ajanı
- 🏗️ Optimize edilmiş [128, 64] Gizli Katman (Hidden Layer) mimarisi
- 🔄 Deneyim Havuzu (Replay Buffer) ile kararlı öğrenme
- 📉 Epsilon-Decay mantığı ile Keşif/Sömürü (Exploration/Exploitation) dengesi

### 📈 Karşılaştırmalı Analiz & Raporlama
- 🏆 Optimizasyon sonucu DQN ile Greedy algoritmaya göre **%36.3** başarı artışı
- 🎯 Dinamik Programlama teorik optimum sınırının **%90.3**'üne ulaşma başarısı
- 📊 Eğitim süreci ve sonuçların otomatik olarak görselleştirilmesi

## 📸 Grafik ve Çıktılar

*(Eğitim tamamlandığında `plots/` klasöründe oluşan grafikler buraya eklenecektir)*
- `reward_curve.png` : DQN Eğitim Eğrisi
- `comparison_bar.png` : Algoritmaların Performans Karşılaştırması

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8 veya üzeri
- Gerekli Python kütüphaneleri (PyTorch, Gymnasium, NumPy, Matplotlib)

### Adımlar

```bash
# 1. Proje klasörüne gidin
cd smart_resource_allocation

# 2. Gerekli kütüphaneleri yükleyin
pip install torch gymnasium numpy matplotlib

# 3. DQN Ajanını Eğitin (Model saved_models/ altına kaydedilir)
python train.py

# 4. Modelleri Karşılaştırın ve Sonuçları Görün
python compare.py

