"""Sentetik verilerle 17 noktalı iskelet yer işareti modelini eğitir (YOLO-pose, COCO ağırlıklarından ince ayar).

Kullanım:
    python 08_nokta_modeli_egit.py                 # NVIDIA ekran kartı varsa onu kullanır
    python 08_nokta_modeli_egit.py --epoch 5 --cpu # hızlı deneme
Çıktı: cikti/nokta_modeli.pt
"""
import argparse
import shutil

from ayarlar import CIKTI, VERI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default=60)
    p.add_argument("--boyut", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--temel", default="yolo11n-pose.pt", help="yolo11s-pose.pt daha doğru ama yavaş")
    a = p.parse_args()

    import torch
    from ultralytics import YOLO

    cihaz = "cpu" if a.cpu or not torch.cuda.is_available() else 0
    print("Cihaz:", "CPU" if cihaz == "cpu" else torch.cuda.get_device_name(0))
    model = YOLO(a.temel)
    sonuc = model.train(data=str(VERI / "sentetik" / "iskelet.yaml"), epochs=a.epoch, imgsz=a.boyut,
                        batch=a.batch, device=cihaz, project=str(CIKTI / "egitim"), name="nokta", exist_ok=True,
                        workers=2 if cihaz == "cpu" else 4, patience=15, fliplr=0.5, degrees=10, scale=0.4,
                        hsv_h=0, hsv_s=0, mosaic=0.3, close_mosaic=5, plots=True, verbose=False)
    en_iyi = CIKTI / "egitim" / "nokta" / "weights" / "best.pt"
    shutil.copy(en_iyi, CIKTI / "nokta_modeli.pt")
    print("Model kaydedildi -> cikti/nokta_modeli.pt")
    m = getattr(sonuc, "results_dict", {})
    if m:
        print("Doğrulama (sentetik) pose mAP50-95:", round(m.get("metrics/mAP50-95(P)", float("nan")), 3))


if __name__ == "__main__":
    main()
