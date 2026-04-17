# Autonomous Driving Project - Dual-Brain V2 & Asymmetric Loss

Bu proje, otonom sürüş görevlerini yerine getirmek (imitation learning) amacıyla [MetaDrive](https://github.com/metadriverse/metadrive) simülatörü üzerinde oluşturulmuştur. Son güncellemelerle proje, basit bir taklitçi olmaktan çıkmış; **Sensör Füzyonu (Sensor Fusion)**, **Çift Beyinli Mimari V2 (Dual-Stream Architecture)** ve **Asimetrik Kayıp Fonksiyonu (Asymmetric Loss)** kullanarak kural tabanlı (rule-based) hiçbir hileye başvurmadan tamamen saf yapay zeka ile sürüş ve acil frenleme (AEB) yapabilen bir seviyeye ulaşmıştır.

## Neler Değişti & Sisteme Neler Eklendi?

### 1. Çift Beyin V2: Bağımsız Tam Görüş (Dual-Stream Architecture)
İlk Çift Beyin denemesinde direksiyon sadece şeride, gaz/fren ise sadece derinliğe bakıyordu. Bu durum "Kör Direksiyon" sorununa yol açtı: Araç önüne araba kırdığında gaz beyni frene basıyor ancak direksiyon beyni engeli göremediği için şerit değiştirmeyi akıl edemiyordu.
* **Çözüm (V2):** `DrivingPolicyNet` mimarisi güncellendi. Direksiyon (Steering Branch) ve Gaz (Acceleration Branch) beyinlerinin ağırlıkları ve nöronları tamamen ayrık tutulmaya devam edildi, ancak **her iki beyne de 2 Kanallı tam giriş (Şerit + Derinlik) verildi.** Böylece direksiyon beyni artık engelleri de görerek çarpışmadan kaçınmak için şerit değiştirmeyi kendi kendine öğrenebilir duruma getirildi.

### 2. Saf Yapay Zeka ile Acil Frenleme: Asimetrik Kayıp Fonksiyonu (Brake Loss Penalty)
İmitasyon öğrenmesinde (Behavioral Cloning) veri setinin %90'ından fazlası gaza basma (pozitif) eylemlerinden oluşur. Standart bir MSE Loss kullanıldığında yapay zeka hata oranını düşük tutmak için "frene basma" eylemini önemsiz bir detay olarak görüp görmezden gelmekte ve engellere bodoslama çarpmaktaydı.
* **Çözüm:** Sistemi `if/else` bloklarıyla zorlamak yerine, kayıp fonksiyonuna (Loss Function) matematiksel bir zeka eklendi. Yazılan `custom_driving_loss` sayesinde; yapay zeka normal yolda hata yaparsa standart 1 birim ceza alırken, **frene basması gereken bir yerde bunu kaçırırsa 3 kat (x3) daha fazla ceza çarpanına** maruz bırakıldı. Bu "Asimetrik Ceza" sistemi sayesinde yapay zeka, kural tabanlı bir müdahaleye gerek kalmadan engelleri gördüğünde frene basmayı bizzat kendi inisiyatifiyle öğrenmiştir.

---

## Önemli Parametreler ve Değişkenler

`src/single_script.py` içerisinde oynayabileceğiniz parametreler:

* **`--episodes` (Veri Toplama):** "Expert" algoritmanın kaç bölümlük örnek veri toplayacağını belirler.
* **`--epochs` (Eğitim Kararı):** Veri setinin baştan sona kaç tur izlenerek eğitileceğini belirler. (Çift beyinli asimetrik loss için 20-30 epoch arası idealdir).
* **`brake_mult = 2.0`:** Eğitim fonksiyonu içindeki frenleme hassasiyeti. Bu değer artırıldıkça araç daha "paranoyak" ve garantici frenler yapar, düşürüldükçe engellere daha çok yaklaşır.
* **`FPS_DIVIDER=1`:** Test aşamasında AI pencerelerinin ekrana yansıma sıklığı. (Örn: 4 yapılırsa görüntü seyrek yenilenir ancak simülasyon FPS'si tavan yapar).

---

## Proje Nasıl Çalıştırılır?

Proje modüler olarak 3 adımdan oluşur: `collect`, `train`, `test`. 

*Not: Ağ mimarisi (Çift Beyin V2) ve Kayıp Fonksiyonu değiştiği için eski ağırlık dosyaları geçersizdir. Önceden toplanmış 2-kanallı füzyon veriniz varsa sadece `--mode train` adımıyla modeli yeniden eğitmeniz yeterlidir.*

### 1- Çevre Verilerini Toplama (Data Collect)
Uzman araç, 400x400 kameradan aldığı görüntüleri anında Şerit Maskesi ve Derinlik Haritasına çevirir, "gürültü (noise)" ekleyerek kurtarma manevralarını diske kaydeder.
```bash
python src/single_script.py --mode collect --episodes 10
```

### 2- Yeni Mimaride Ağı Eğitme (Train)
`dataset` içerisindeki veriler kullanılarak Asimetrik Kayıp Fonksiyonu ile Direksiyon ve Gaz/Fren ağları eğitilir.
```bash
python src/single_script.py --mode train --epochs 20
```

### 3- Otonom Test (Test)
Model gerçek zamanlı simülasyonla test edilir. AI Derinlik ve Şerit gözleri ekrandan canlı takip edilebilir.
```bash
python src/single_script.py --mode test
```