from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Ultralytics'in kendi örnek resmini kullanalım (internetten otomatik çeker)
sonuclar = model('https://ultralytics.com/images/bus.jpg')

sonuclar[0].save('test_foto_sonuc.jpg')
print("Tespit tamamlandı! test_foto_sonuc.jpg dosyasına bak.")

for kutu in sonuclar[0].boxes:
    sinif_id = int(kutu.cls[0])
    sinif_adi = model.names[sinif_id]
    guven = float(kutu.conf[0])
    print(f"Tespit edildi: {sinif_adi} (güven: %{guven*100:.1f})")