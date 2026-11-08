"""Yer işareti modelini eğitmek için BT ölçülü sentetik iskelet çizimleri üretir (YOLO-pose biçiminde).

Kullanım:
    python 07_sentetik_veri.py --egitim 6000 --dogrulama 600
Çıktı: veri/sentetik/{images,labels}/{train,val} ve veri/sentetik/iskelet.yaml
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ayarlar import CIKTI, VERI
import sentetik_iskelet as si

HEDEF = VERI / "sentetik"
BOYUT = (512, 768)


def uret(arg):
    bolum, i, tohum, model_yolu = arg
    rng = np.random.default_rng(tohum)
    model = si.model_oku(model_yolu) if model_yolu else None
    img, figurler = si.ornek_coklu(rng, model, BOYUT)
    W, H = BOYUT
    ad = f"{bolum}_{i:06d}"
    img.convert("RGB").save(HEDEF / "images" / bolum / f"{ad}.jpg", quality=int(rng.integers(70, 96)))
    satirlar = []
    for kp, (mn, mx) in figurler:
        mn = np.clip(mn - 10, 0, [W, H])
        mx = np.clip(mx + 10, 0, [W, H])
        if (mx - mn).min() < 20:
            continue
        cx, cy = (mn + mx) / 2 / [W, H]
        w, h = (mx - mn) / [W, H]
        parca = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
        for k in si.NOKTA_SIRASI:
            x, y = kp[k]
            gorunur = 2 if 0 <= x < W and 0 <= y < H else 0
            parca.append(f"{x / W:.6f} {y / H:.6f} {gorunur}" if gorunur else "0 0 0")
        satirlar.append(" ".join(parca))
    (HEDEF / "labels" / bolum / f"{ad}.txt").write_text("\n".join(satirlar) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--egitim", type=int, default=6000)
    p.add_argument("--dogrulama", type=int, default=600)
    p.add_argument("--varsayilan-olcu", action="store_true", help="BT modeli yerine varsayılan ölçüler")
    a = p.parse_args()
    model_yolu = None if a.varsayilan_olcu or not (CIKTI / "bt_model.json").exists() else str(CIKTI / "bt_model.json")
    print("Ölçüler:", "BT modeli (cikti/bt_model.json)" if model_yolu else "varsayılan")
    for b in ("train", "val"):
        (HEDEF / "images" / b).mkdir(parents=True, exist_ok=True)
        (HEDEF / "labels" / b).mkdir(parents=True, exist_ok=True)
    isler = [("train", i, 1000 + i, model_yolu) for i in range(a.egitim)] + \
            [("val", i, 900000 + i, model_yolu) for i in range(a.dogrulama)]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as havuz:
        for j, _ in enumerate(havuz.map(uret, isler, chunksize=16), 1):
            if j % 500 == 0:
                print(f"  {j}/{len(isler)}")
    yaml = (f"path: {HEDEF.as_posix()}\ntrain: images/train\nval: images/val\n"
            f"kpt_shape: [17, 3]\nflip_idx: {si.FLIP_IDX}\nnames:\n  0: iskelet\n")
    (HEDEF / "iskelet.yaml").write_text(yaml)
    print(f"Bitti -> {HEDEF}")


if __name__ == "__main__":
    main()
