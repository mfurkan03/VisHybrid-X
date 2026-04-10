# Autonomous Driving Project

Bu proje, otonom sürüş görevlerini yerine getirmek (imitation learning) amacıyla [MetaDrive](https://github.com/metadriverse/metadrive) simülatörü üzerinde oluşturulmuştur. Son yapılan köklü güncellemeler ile projenin çevreyi analiz etme (kodlama/görme) yeteneği tamamen gerçekçi bir Derinlik Haritası (Depth Map) mimarisi üzerine inşa edilmiştir.

## Neler Değişti & Neden Baştan "Train" Ettik?

### 1. Pseudo-Depth (Sahte Derinlik) Yerine Gerçek "Depth Anything" Adaptasyonu
Daha önceden sistem, kamera ortamından (MetaDrive) aldığı renkli (RGB) görüntülerin sadece matematiksel olarak renk ortalamasını (Grayscale/Siyah-Beyaz) alıp bunu yapay zekaya "derinlik" olarak yutturuyordu. Bu aldatmaca, yapay zekanın yoldaki silüetleri tanıyabiliyormuş gibi gözükmesine rağmen aslında nesnelerin derinliğini (gökyüzü boşluğu mu yoksa beton duvar mı olduğunu) algılayamamasına sebep oluyordu.

Biz projeye gerçek zamanlı ve stabil derinlik tespiti için **Video Depth Anything** isimli son teknoloji (Foundation Model) bir mimariyi harici olarak entegre ettik.
Mevcut ağı baştan eğitmemizin (Train etmemizin) temel sebebi şuydu: Modelimiz `(1, 84, 84)` boyutlarında girdiler alan basit sinir ağı yapısını değiştirmedi ancak ona verilen girdi kökten farklılaştı. Yapay zeka artık siyah-beyaz fotolar yerine gerçek ve piksellerin pürüzsüz karanlıklarına göre derinlikleri bildiren ("yakın olan parlak, uzak olan bölgeler koyu") haritalara göre sürmeyi öğrenmek zorundaydı.

### 2. Test Ekranında Gerçek Görüntü (Visualization)
Model, arka planda muazzam bir şekilde gerçek derinlik haritalarını görüp ona göre sürmesine rağmen test ortamında sol pencerede ham "RGB" görüntüsü basan bir kod kalmıştı. Yapılan değişiklikle, yapay zekanın retinasına giren "Asıl Derinlik Haritası", **`INFERNO` (Termal/Isı)** renk haritası paleti kullanılarak insan gözünün göreceği şekilde ekrana (Test Asamasi - AI) yansıtıldı.

### 3. FPS ve Hızlandırma Optimizasyonları (Streaming Modu)
İlk denemelerde test aşaması (otonom aracın sürüş aşaması) inanılmaz ağırlaşmıştı (0.4 FPS). Bunun nedeni `Video-Depth-Anything` modelinin videoları stabilizesi yüksek şekilde idrak edebilmek için doğası gereği GPU'da her seferinde **Entegre 32-Karelik** kayan bir video matrisi hesaplamasıdır. Araba bir saniyede 5 kare gönderse dahi model bunları 32 kare gibi doldurarak sistemi boğuyordu.

**Optimizasyon Çözümü:** Otonom ajanın sistemi, kaba hesaplama döngüsünden çıkarıldı ve doğrudan Caching mekanizmasına sahip olan `VideoDepthAnythingStream` (Streaming Modu) altyapısına bağlandı. Bu mod, aracın akışındaki önceki 31 karenin özelliklerini saniyelerce hesaplamak yerine onları anında belleğinde (*cache*) tutar. Ağa sadece kameradan o an gelen "1 tekil karenin" özelliği (feature) verilir, eski hafıza ile bir saniyenin küçük bir kısmında kaydırılıp mükemmel bir sonuç çıkartılır.
***Sonuç:*** Araç kilitlenmeleri saniyelerden milisaniyelere düştü ve sistem GPU kullanırken anlık **20+ FPS** akıcılıkla sürüş yapma hızına kavuştu!

---

## Önemli Parametreler ve Değişkenler

`src/single_script.py` içerisinde oynayabileceğiniz veya komut satırından dinamik olarak atayabileceğiniz parametreler şunlardır:

* **`--episodes` (Veri Toplama için):** Otonom aracın "Expert" algoritmada süreceği ve kaç bölümlük örnek veri toplayacağını (`collect` modunda) belirler. (Daha yüksek bölüm daha stabil eğitimi sağlar ancak hafıza kaplar.)
* **`--epochs` (Eğitim Kararı):** Yapay zekamızın (`DrivingPolicyNet`) oluşturulan `dataset` içindeki npz uzantılı verileri baştan sona kaç tur izleyerek eğiteceğini belirler. (Genellikle 20 idealdir).
* **`encoder` (`vits` / `vitb` / `vitl`):** Kodda oluşturulan `DepthEstimationModel` içindeki yapay zekanın devasa parametre büyüklüğüdür. `vitb` ve `vitl` modelleri aşırı VRAM isteyeceğinden, bilgisayar optimizasyonu ve hız için en ideal olan **`vits`** modeli ayarlanmıştır.
* **`input_size=252`:** Derinlik (VDA) tespiti esnasında sinir ağlarına girecek piksellenme işlem boyutudur. Kesinlikle 14'ün katı olmalıdır. Eğer bunu örn: 112 veya 140'lara kadar düşürüp FPS'yi daha da katlamaya çalışırsanız; DINOv2 ağındaki otonomik dikkat (attention) mekanizmaları yapıyı algılayamayacak kadar minyatür göreceği için size derinlik haritası yerine `NaN` (geçersiz, çökmüş bozuk matrix) fırlatmaya başlar. Bu yüzden kalite ve hızın mükemmel ortalaması olan `252`'ye optimize edilmiştir.
* **`target_fps=30`:** VDA Stream modelinin kendi içindeki saniye tutarlılık oranıdır, değiştirmeyin.

---

## Proje Nasıl Çalıştırılır?

Çalışma ortamımız modüler bir biçimde 3 farklı pipeline'ın tek dosyada buluştuğu (`collect`, `train`, `test`) komut mimarisine sahiptir. İsterseniz `--mode all` kullanabilirsiniz.
*Lütfen kendi Python Sanal Ortamınızın (venv) ve GPU Cuda sürümünüzün (Örn: pyTorch cu118) uyumlu yüklendiğinden emin olun.*

### 1- Çevre Verilerini Toplama (Data Collect)
Expert sistemin yola çıkıp çevreyi süzdüğü aşama. Simülatör çalışır ve yoldaki bütün RGB fotoğrafları (hiç donma yaşamadan) biriktirir, aracın macerası bittiği saniye o "Sürüş Videosunu" VDA `predict_batch`'ine toplu fırlatıp hepsinin derinlik matrislerini tek seferde çıkararak `dataset` klasörüne yazar. (Sistemi asla boğmaz).
```bash
python src/single_script.py --mode collect --episodes 10
```

### 2- Yeni Modele Göre Ağı (Beyni) Eğitme (Train)
`dataset` içerisine çıkarılmış üst kalite derinlik haritalarını alıp sinir ağımızın (`DrivingPolicyNet`) derinlik tabanlı engelleri aşmayı öğrenmesi evresidir.
```bash
python src/single_script.py --mode train --epochs 20
```

### 3- Yapay Zeka Testini (Otonom) Çalıştırma (Test)
Bizim eğittiğimiz sistem gerçek dünya koşullarıyla test edilir! Araç çalışırken her saniye VDA Stream sayesinde anlık kameralar Isı Temalı (Inferno) bir derinlik (Depth) modellemesine dönüştürüp, ağınıza yedirilir ve araba harikalar yaratır.
```bash
python src/single_script.py --mode test
```
