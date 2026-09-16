import cv2
import numpy as np

# 300x300 boyutunda, siyah bir resim oluştur
resim = np.zeros((300, 300, 3), dtype=np.uint8)

# Üzerine yeşil bir daire çiz (merkez: 150,150, yarıçap: 100)
cv2.circle(resim, (150, 150), 100, (0, 255, 0), -1)

# Resmi bir dosyaya kaydet
cv2.imwrite("test_resim.png", resim)

print("Resim oluşturuldu ve kaydedildi!")