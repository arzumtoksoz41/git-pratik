## MNIST El Yazısı Rakam Sınıflandırıcı

PyTorch'a giriş projem. İki farklı model mimarisini (basit sinir ağı ve CNN)
karşılaştırarak, konvolüsyonel katmanların görüntü işlemedeki etkisini inceledim.

### Sonuçlar

| Model | Test Doğruluğu |
|---|---|
| Basit Sinir Ağı (Fully Connected) | %96.67 |
| CNN (Convolutional Neural Network) | **%98.72** |

**Gözlem:** CNN, konvolüsyon katmanları sayesinde piksellerin konumsal
(spatial) ilişkisini koruyabildiği için, basit modele göre daha yüksek
doğruluk ve daha istikrarlı bir eğitim süreci (kayıp değerleri daha düzenli
azaldı) gösterdi.

### Kullanılan yöntemler
- PyTorch tensor ve otomatik gradyan hesaplama (autograd)
- `nn.Sequential` ile iki farklı model mimarisi
- Conv2d, MaxPool2d katmanları (CNN mimarisi)
- Adam optimizer ve CrossEntropyLoss kayıp fonksiyonu
- Eğitim/test verisi ayrımı ile gerçek performans ölçümü

**Neden yaptım:** PyTorch'un temel kavramlarını (tensor, gradyan, epoch) ve
görüntü işlemede neden CNN mimarisinin tercih edildiğini uygulamalı olarak
öğrenmek için. Bu, ileride görüntü işleme ve otonom sistemler alanında
yapacağım İHA/drone projeleri için bir başlangıç noktası — YOLO gibi
nesne tespiti modelleri de CNN temelli çalışıyor.