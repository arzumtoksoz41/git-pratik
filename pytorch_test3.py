import torch

# Gerçek kural şu olsun (biz biliyoruz ama modele söylemiyoruz): y = 2*x
# Modelin bunu ÖRNEKLERDEN kendi bulmasını istiyoruz

x_verileri = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_verileri = torch.tensor([2.0, 4.0, 6.0, 8.0])   # gerçek sonuçlar (y = 2*x)

# Modelin "tahmini" - başta rastgele bir sayı, mesela 0.0'dan başlasın
agirlik = torch.tensor(0.0, requires_grad=True)

# 20 kere "dene, hatanı gör, düzelt" döngüsü
for adim in range(20):
    tahmin = agirlik * x_verileri          # modelin şu anki tahmini
    hata = ((tahmin - y_verileri) ** 2).mean()   # ne kadar yanlış olduğu

    hata.backward()   # gradyanı hesapla ("hangi yöne düzelmeliyim")

    with torch.no_grad():
        agirlik -= 0.1 * agirlik.grad   # küçük bir adım at, düzelt
        agirlik.grad.zero_()             # bir sonraki tur için sıfırla

    if adim % 5 == 0:
        print(f"Adım {adim}: agirlik = {agirlik.item():.3f}, hata = {hata.item():.3f}")

print(f"\nSonuç: Model 'agirlik'in {agirlik.item():.2f} olduğunu öğrendi (gerçek değer: 2.0)")