import torch
import torchvision
import torchvision.transforms as transforms

# MNIST veri setini indir (ilk seferde internetten çeker, biraz sürebilir)
donusum = transforms.ToTensor()
egitim_verisi = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=donusum)

print(f"Toplam eğitim resmi sayısı: {len(egitim_verisi)}")

# İlk resmi ve etiketini incele
resim, etiket = egitim_verisi[0]
print(f"İlk resmin boyutu: {resim.shape}")
print(f"İlk resmin gerçek rakamı (etiket): {etiket}")