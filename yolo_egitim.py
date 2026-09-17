from ultralytics import YOLO

# Sıfırdan değil, küçük bir temel model üzerine "fine-tune" yapacağız
model = YOLO('yolov8n.pt')

# COCO128 küçük örnek veri setiyle eğit (otomatik indirilir)
sonuc = model.train(data='coco128.yaml', epochs=5, imgsz=640)

print("Eğitim tamamlandı!")