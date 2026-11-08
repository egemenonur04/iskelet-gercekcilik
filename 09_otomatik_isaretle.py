"""Eğitilmiş yer işareti modeliyle görselleri insan girdisi olmadan işaretler.

Çıktı: isaretler/otomatik.json (işaretleme aracıyla aynı biçim; araçta açıp kontrol edilebilir)
Elle işaretlenmiş JSON'lar da varsa model ile insan arasındaki uyumu raporlar:
cikti/nokta_uyumu.csv ve cikti/nokta_uyumu.md
"""
import argparse
import json
from datetime import datetime

import numpy as np

from ayarlar import CIKTI, GORSELLER, ISARETLER
from sentetik_iskelet import NOKTA_SIRASI

UZANTI = {".png", ".jpg", ".jpeg", ".webp"}


def isaretle(model, yol, esik):
    r = model.predict(str(yol), verbose=False, conf=0.05, imgsz=640)[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    kutu = r.boxes.xywh.cpu().numpy()
    guven = r.boxes.conf.cpu().numpy()
    H, W = r.orig_shape
    merkez = 1 - np.hypot(kutu[:, 0] / W - 0.5, kutu[:, 1] / H - 0.5)
    sec = int(np.argmax(guven * kutu[:, 2] * kutu[:, 3] * merkez))   # ana figür: güvenli, büyük, ortada
    xy = r.keypoints.xy[sec].cpu().numpy()
    kc = r.keypoints.conf[sec].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
    noktalar, atlanan, guvenler = {}, [], {}
    for ad, (x, y), c in zip(NOKTA_SIRASI, xy, kc):
        guvenler[ad] = round(float(c), 3)
        if c >= esik:
            noktalar[ad] = [round(float(x), 1), round(float(y), 1)]
        else:
            atlanan.append(ad)
    return {"dosya": yol.name, "noktalar": noktalar, "atlanan": atlanan, "nokta_guveni": guvenler,
            "figur_guveni": round(float(guven[sec]), 3), "genislik": W, "yukseklik": H,
            "kaburga_a": None, "kaburga_b": None, "lomber": None, "parmak_el": None,
            "talimat_disi": False, "not": "otomatik", "kaynak_isaret": "model"}


def uyum_raporu(oto):
    insan = {}
    for yol in ISARETLER.glob("*.json"):
        if yol.name.startswith("otomatik"):
            continue
        for g in json.loads(yol.read_text(encoding="utf-8")).get("gorseller", []):
            insan[g["dosya"]] = g
    satirlar = []
    for g in oto:
        h = insan.get(g["dosya"])
        if not h:
            continue
        n_i, n_o = h.get("noktalar", {}), g["noktalar"]
        if "kafatasi_1" not in n_i or "kafatasi_2" not in n_i:
            continue
        olcek = np.hypot(*np.subtract(n_i["kafatasi_1"], n_i["kafatasi_2"]))
        for ad in NOKTA_SIRASI:
            if ad in n_i and ad in n_o:
                hata = np.hypot(*np.subtract(n_i[ad], n_o[ad]))
                # a/b tarafı karışmışsa diğer tarafla da dene
                es = ad[:-1] + ("b" if ad.endswith("a") else "a") if ad.endswith(("_a", "_b")) else None
                if es and es in n_o:
                    hata = min(hata, np.hypot(*np.subtract(n_i[ad], n_o[es])))
                satirlar.append({"dosya": g["dosya"], "nokta": ad, "hata_px": hata, "hata_kafatasi": hata / olcek})
    if not satirlar:
        return
    import pandas as pd
    df = pd.DataFrame(satirlar)
    df.to_csv(CIKTI / "nokta_uyumu.csv", index=False)
    ozet = df.groupby("nokta").hata_kafatasi.agg(["count", "median", lambda x: (x < 0.5).mean()])
    ozet.columns = ["n", "medyan hata (kafatası çapı)", "PCK@0,5"]
    r = ["# Otomatik ve elle işaretleme uyumu", "",
         f"{df.dosya.nunique()} görselde {len(df)} nokta karşılaştırıldı.",
         "Hata, kafatası çapına bölünmüştür. PCK@0,5: hatası kafatası çapının yarısından küçük noktaların oranı.", "",
         ozet.round(3).to_markdown()]
    (CIKTI / "nokta_uyumu.md").write_text("\n".join(r), encoding="utf-8")
    print(ozet.round(3).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esik", type=float, default=0.3, help="Bu güvenin altındaki noktalar 'atlandı' sayılır")
    a = p.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(CIKTI / "nokta_modeli.pt"))
    dosyalar = sorted(f for k in ("ogrenci", "yz") for f in (GORSELLER / k).glob("*") if f.suffix.lower() in UZANTI)
    sonuc = []
    for f in dosyalar:
        s = isaretle(model, f, a.esik)
        if s:
            sonuc.append(s)
            print(f"{f.name}: {len(s['noktalar'])}/17 nokta, figür güveni {s['figur_guveni']}")
        else:
            print(f"{f.name}: iskelet bulunamadı")
    (ISARETLER / "otomatik.json").write_text(json.dumps(
        {"surum": 1, "olusturma": datetime.now().isoformat(), "kaynak": "otomatik", "noktalar": NOKTA_SIRASI,
         "gorseller": sonuc}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(sonuc)} görsel işaretlendi -> isaretler/otomatik.json")
    uyum_raporu(sonuc)


if __name__ == "__main__":
    main()
