import cv2

# Gri resmi oku
gri_resim = cv2.imread("test_resim_gri.png", cv2.IMREAD_GRAYSCALE)

# Kenarları bul (Canny algoritması)
kenarlar = cv2.Canny(gri_resim, 50, 150)

# Sonucu kaydet
cv2.imwrite("test_resim_kenarlar.png", kenarlar)

print("Kenar tespiti tamamlandı!")