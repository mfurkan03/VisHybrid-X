# Autonomous Driving Project - Sensor Fusion & Dual Architecture

Bu proje, otonom sürüş görevlerini yerine getirmek (imitation learning) amacıyla [MetaDrive](https://github.com/metadriverse/metadrive) simülatörü üzerinde oluşturulmuştur. Son yapılan köklü güncellemeler ile proje; sadece "Derinlik Haritası"na bakan tek boyutlu bir yapıdan, "Derinlik + Şerit Takibi" yapabilen **Sensör Füzyonu (Sensor Fusion)** tabanlı **Çift Beyinli (Dual-Stream)** bir yapay zeka mimarisine evrilmiştir.

## Neler Değişti & Sisteme Neler Eklendi?

### 1. Şerit Takibi (Lane Masking) ve Sensör Füzyonu
Sadece Derinlik Haritası (Depth Map) kullanmak, yapay zekanın engelleri ve diğer araçları görmesini sağlasa da; asfalt ile yol dışındaki çimenlik alanın kameraya olan uzaklığı aynı olduğu için aracın sürekli yoldan çıkmasına (out_of_road) sebep oluyordu.
* **Çözüm:** Kamera çözünürlüğü 84x84'ten 400x400'e çıkarılarak yüksek çözünürlüklü RGB görüntüler üzerinden OpenCV ile **ROI (Region of Interest)** ve **Thresholding (Eşikleme)** işlemleri uygulandı. Gökyüzü ve binalar maskelenerek yoldaki şerit çizgileri siyah-beyaz net bir matrise dönüştürüldü.
* **Sensör Füzyonu:** Elde edilen bu *Şerit Maskesi*, *Derinlik Haritası* ile üst üste bindirilerek modelin girdisi 1 kanaldan **2 kanala** (Channel) çıkarıldı. Yapay zeka artık hem fiziksel derinliği hem de yoldaki boyaları aynı anda görebilmektedir.

### 2. RAM (OOM) ve Veri Depolama Optimizasyonu (On-the-Fly Processing)
Yüksek çözünürlüklü (400x400) kameralara geçilmesiyle birlikte, veri toplama (`collect`) aşamasında ham RGB resimlerinin RAM'de listelenmesi sistemin çökmesine (Killed / Out of Memory) sebep olmaktaydı.
* **Çözüm:** Görüntüler artık listelerde bekletilmek yerine simülasyondan alındığı **an (on-the-fly)** işlenerek şerit ve derinlik haritaları çıkartılıp anında 84x84 boyutlarına küçültülmektedir. Eski RGB verileri diske kaydedilmekten çıkarılmış, böylece hem RAM şişmesi tamamen engellenmiş hem de `.npz` veri setlerinin boyutu Gigabaytlardan Megabaytlara düşürülerek muazzam bir hız kazanılmıştır.

### 3. "Çift Beyinli" Sinir Ağı Mimarisi (Dual-Stream Architecture)
Modelin tek bir ortak sinir ağı üzerinden hem gaz/fren hem de direksiyon kararı vermesi literatürde **Görev Çakışması (Task Interference)** olarak bilinen soruna yol açıyordu. Yapay zeka engellerden (derinlikten) kaçmaya odaklandığında şeritleri okumayı unutuyor, şeritleri okumaya çalıştığında frene basmaya korkuyordu.
* **Çözüm:** `DrivingPolicyNet` mimarisi kökten değiştirildi. Ortak havuz ikiye bölündü:
  1. **Şerit Beyni (Steering Branch):** Sadece şerit maskesini (1. Kanal) okuyarak direksiyon kırma eylemini hesaplar.
  2. **Gaz/Fren Beyni (Acceleration Branch):** Sadece derinlik haritasını (0. Kanal) okuyarak engelleri fark edip hızlanma veya acil frenleme eylemini hesaplar.
Bu sayede her iki beyin birbirinin ağırlıklarını (gradient) bozmadan kendi görevinde uzmanlaşmıştır.

### 4. VDA Cache (Hafıza) Sıfırlama Düzeltmesi
Video Depth Anything modelinin Streaming modunda önceki kareleri (cache) aklında tutması özelliği, simülasyonda yeni bir bölüme (episode) geçildiğinde aracın önceki bölümdeki kaza anını hatırlayıp aniden duvara kırmasına sebep oluyordu. Sisteme her bölüm başında VDA modelini sıfırlayan (Cache Reset) bir kod eklenerek bu "hafıza kayması" sorunu kökünden çözülmüştür.

---

## Önemli Parametreler ve Değişkenler

`src/single_script.py` içerisinde oynayabileceğiniz veya komut satırından dinamik olarak atayabileceğiniz parametreler şunlardır:

* **`--episodes` (Veri Toplama için):** Otonom aracın "Expert" algoritmada süreceği ve kaç bölümlük örnek veri toplayacağını (`collect` modunda) belirler.
* **`--epochs` (Eğitim Kararı):** Yapay zekamızın (`DrivingPolicyNet`) oluşturulan `dataset` içindeki npz uzantılı verileri baştan sona kaç tur izleyerek eğiteceğini belirler. (Çift beyinli mimari için 20 epoch idealdir).
* **`encoder` (`vits`):** Kodda oluşturulan `DepthEstimationModel` içindeki yapay zekanın devasa parametre büyüklüğüdür. Bilgisayar optimizasyonu ve hız için en ideal olan **`vits`** modeli ayarlanmıştır.
* **`FPS_DIVIDER=1`:** Test aşamasında yapay zekanın "Şerit" ve "Derinlik" gözlerinin ekrana ne sıklıkla renderlanacağını belirler. Performans darboğazı yaşanırsa bu değeri 3 veya 4 yaparak sistem FPS'sini uçurabilirsiniz.

---

## Proje Nasıl Çalıştırılır?

Çalışma ortamımız modüler bir biçimde 3 farklı pipeline'ın tek dosyada buluştuğu (`collect`, `train`, `test`) komut mimarisine sahiptir.
*Not: Ağ mimarisi (Çift Beyin) ve girdi kanalları (2 Channel) kökten değiştiği için eski modeller ve eski veri setleri ile çalışmaz. Tüm adımların sıfırdan uygulanması gerekir.*

### 1- Çevre Verilerini Toplama (Data Collect)
Expert sistem yola çıkar, 400x400 kameradan aldığı görüntüleri anında Şerit Maskesine ve Derinlik Haritasına çevirip birleştirerek RAM dostu bir şekilde kaydeder. *(Ayrıca sisteme eklenen ufak gürültüler (noise) ile uzman botun aracı kurtarma manevraları da veri setine eklenir).*
```bash
python src/single_script.py --mode collect --episodes 10
```

### 2- Çift Beyinli Ağı Eğitme (Train)
`dataset` içerisine çıkarılmış 2-kanallı füzyon verilerini alıp sinir ağımızın (`DrivingPolicyNet`) direksiyon ve gaz/fren kollarını (branch) ayrı ayrı eğittiği evredir.
```bash
python src/single_script.py --mode train --epochs 20
```

### 3- Yapay Zeka Testini (Otonom) Çalıştırma (Test)
Kendi eğittiğimiz sistem gerçek dünya koşullarıyla test edilir! Ekranda 3 farklı pencere açılır: Ana RGB Kamera, AI Derinlik Gözü (Inferno Isı Haritası) ve AI Şerit Gözü. Aracımız çift beyni ile virajları kendi alır, engelleri kendi aşar.
```bash
python src/single_script.py --mode test
```