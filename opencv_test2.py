import cv2

# Az önce oluşturduğumuz resmi oku
resim = cv2.imread("test_resim.png")

# Resmin boyutlarını yazdır (yükseklik, genişlik, kanal sayısı)
print("Resim boyutu:", resim.shape)

# Resmi gri tonlamaya çevir
gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

# Gri resmi kaydet
cv2.imwrite("test_resim_gri.png", gri_resim)

print("Gri tonlamalı resim kaydedildi!")