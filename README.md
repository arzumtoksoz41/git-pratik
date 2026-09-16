## MNIST El Yazısı Rakam Sınıflandırıcı

PyTorch'a giriş projem. Basit bir sinir ağı (fully connected neural network) ile
MNIST veri setindeki el yazısı rakamları (0-9) sınıflandırıyor.

**Sonuç:** Test verisi üzerinde %96.67 doğruluk.

**Kullanılan yöntemler:**
- PyTorch tensor ve otomatik gradyan hesaplama (autograd)
- `nn.Sequential` ile basit bir sinir ağı mimarisi (Flatten → Linear → ReLU → Linear)
- Adam optimizer ve CrossEntropyLoss kayıp fonksiyonu
- Eğitim/test verisi ayrımı ile gerçek performans ölçümü

**Neden yaptım:** PyTorch'un temel kavramlarını (tensor, gradyan, epoch, eğitim döngüsü)
uygulamalı olarak öğrenmek için. Bu, ileride görüntü işleme ve otonom sistemler
alanında yapacağım projeler için bir başlangıç noktası.

