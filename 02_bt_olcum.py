"""BT kemik maskelerinden 3B yer işaretleri ve uzunlukları hesaplar.

Çıktılar:
    cikti/bt_olcumler.csv        her denek için uzunluklar (mm); kesik kemikler boş bırakılır
    cikti/bt_yer_isaretleri.json her denek için 3B noktalar (doğrulama simülasyonunda kullanılır)
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

from ayarlar import CIKTI, MASKE_KLASORU, VERTEBRALAR

MAKS_NOKTA = 40000
# Erişkin anatomisi için geniş olabilirlik aralıkları (mm). BT'nin dairesel görüntü alanı dışında kalan
# kemik parçaları dizi sınırına değmediği için kesik olarak algılanamaz; bu aralıklar onları ayıklar.
ARALIK = {"kafatasi_cap1": (160, 250), "kafatasi_cap2": (115, 180), "kafatasi_cap3": (90, 175), "humerus": (230, 400), "femur": (330, 560),
          "juguler_omuz": (110, 240), "omurga_C7_L5": (330, 560), "kalca_genisligi": (130, 230)}


def aralikta(ad, deger):
    for k, (alt, ust) in ARALIK.items():
        if ad == k or ad.startswith(k + "_"):
            return alt <= deger <= ust
    return True
UC_DILIM = 0.08          # uzun kemik uçlarını tanımlamak için eksen boyunca alınan dilim oranı


def noktalar(yol):
    """Maskeyi mm cinsinden nokta bulutuna çevirir. Maske görüntü sınırına değiyorsa kesik sayılır."""
    if not yol.exists():
        return None, None
    img = nib.load(str(yol))
    veri = np.asanyarray(img.dataobj)
    ijk = np.argwhere(veri > 0)
    if len(ijk) < 50:
        return None, None
    sekil = np.array(veri.shape)
    kesik = bool((ijk.min(0) <= 2).any() or (ijk.max(0) >= sekil - 3).any())
    if len(ijk) > MAKS_NOKTA:
        ijk = ijk[np.random.default_rng(0).choice(len(ijk), MAKS_NOKTA, replace=False)]
    xyz = nib.affines.apply_affine(img.affine, ijk)
    return xyz, kesik


def ana_eksenler(xyz):
    merkez = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - merkez, full_matrices=False)
    izd = (xyz - merkez) @ vt.T
    return merkez, vt, izd.max(0) - izd.min(0), izd


def uzun_kemik(xyz, proksimal_ipucu, medial_ipucu=None):
    """Uzun kemiğin iki eklem ucunu bulur. proksimal_ipucu: proksimal uca yakın olması beklenen 3B nokta."""
    merkez, vt, uzunluklar, izd = ana_eksenler(xyz)
    t = izd[:, 0]
    alt, ust = np.quantile(t, [UC_DILIM, 1 - UC_DILIM])
    uc1, uc2 = xyz[t <= alt], xyz[t >= ust]
    m1, m2 = uc1.mean(0), uc2.mean(0)
    if np.linalg.norm(m1 - proksimal_ipucu) > np.linalg.norm(m2 - proksimal_ipucu):
        uc1, uc2, m1, m2 = uc2, uc1, m2, m1
    if medial_ipucu is not None:   # femur başı: proksimal ucun medial yarısı (trokanteri dışarıda bırakır)
        d = np.abs(uc1[:, 0] - medial_ipucu[0])
        m1 = uc1[d <= np.median(d)].mean(0)
    return m1, m2, float(uzunluklar[0])


def denek_olc(klasor):
    yol = lambda ad: klasor / f"{ad}.nii.gz"
    sonuc, isaret = {"denek": klasor.name}, {}

    kafa, kesik = noktalar(yol("skull"))
    if kafa is not None and not kesik:
        _, _, uz, _ = ana_eksenler(kafa)
        sonuc["kafatasi_cap1"], sonuc["kafatasi_cap2"], sonuc["kafatasi_cap3"] = map(float, uz)

    sternum, kesik = noktalar(yol("sternum"))
    if sternum is not None and not kesik:
        isaret["juguler"] = sternum[sternum[:, 2] >= np.quantile(sternum[:, 2], 0.98)].mean(0)

    # omurga yayı
    merkezler = {}
    for v in VERTEBRALAR:
        p, kesik = noktalar(yol(f"vertebrae_{v}"))
        if p is not None and not kesik:
            merkezler[v] = p.mean(0)
    if "C7" in merkezler:
        isaret["C7"] = merkezler["C7"]
    if "L5" in merkezler:
        isaret["L5"] = merkezler["L5"]
    if all(v in merkezler for v in VERTEBRALAR):
        dizi = np.array([merkezler[v] for v in VERTEBRALAR])
        sonuc["omurga_C7_L5"] = float(np.linalg.norm(np.diff(dizi, axis=0), axis=1).sum())

    kalca, kalca_nokta = {}, {}
    for taraf in ("left", "right"):
        p, _ = noktalar(yol(f"hip_{taraf}"))
        if p is not None:
            kalca[taraf] = p.mean(0)
            kalca_nokta[taraf] = p
    orta_x = np.mean([k[0] for k in kalca.values()]) if len(kalca) == 2 else None
    if orta_x is None and merkezler:
        orta_x = float(np.mean([m[0] for m in merkezler.values()]))

    tr = {"left": "sol", "right": "sag"}
    for taraf in ("left", "right"):
        # humerus
        hum, kesik = noktalar(yol(f"humerus_{taraf}"))
        if hum is not None and not kesik:
            sk, _ = noktalar(yol(f"scapula_{taraf}"))
            ipucu = sk.mean(0) if sk is not None else hum[hum[:, 2].argmax()]
            omuz, dirsek, uz = uzun_kemik(hum, ipucu)
            isaret[f"omuz_{tr[taraf]}"], isaret[f"dirsek_{tr[taraf]}"] = omuz, dirsek
            sonuc[f"humerus_{tr[taraf]}"] = float(np.linalg.norm(omuz - dirsek))
            if "juguler" in isaret:
                sonuc[f"juguler_omuz_{tr[taraf]}"] = float(np.linalg.norm(isaret["juguler"] - omuz))
        # femur
        fem, kesik = noktalar(yol(f"femur_{taraf}"))
        if fem is not None and not kesik:
            ipucu = kalca.get(taraf, fem[fem[:, 2].argmax()])
            medial = np.array([orta_x, 0, 0]) if orta_x is not None else None
            bas, diz, uz = uzun_kemik(fem, ipucu, medial)
            if taraf in kalca_nokta:   # femur başı: asetabuluma en yakın femur noktalarının çevresi
                from scipy.spatial import cKDTree
                d, _ = cKDTree(kalca_nokta[taraf][::3]).query(fem, k=1)
                temas = fem[np.argmin(d)]
                yakin = fem[np.linalg.norm(fem - temas, axis=1) < 30]
                if len(yakin) > 20:
                    bas = yakin.mean(0)
            isaret[f"kalca_{tr[taraf]}"], isaret[f"diz_{tr[taraf]}"] = bas, diz
            sonuc[f"femur_{tr[taraf]}"] = float(np.linalg.norm(bas - diz))
    if "kalca_sol" in isaret and "kalca_sag" in isaret:
        sonuc["kalca_genisligi"] = float(np.linalg.norm(isaret["kalca_sol"] - isaret["kalca_sag"]))

    for k in list(sonuc):
        if k != "denek" and not aralikta(k, sonuc[k]):
            sonuc[k + "_aralik_disi"] = sonuc.pop(k)
    if not all(f"kafatasi_cap{i}" in sonuc for i in (1, 2, 3)):   # kafatası ya tam ya hiç
        for i in (1, 2, 3):
            if f"kafatasi_cap{i}" in sonuc:
                sonuc[f"kafatasi_cap{i}_aralik_disi"] = sonuc.pop(f"kafatasi_cap{i}")
    return sonuc, {k: v.tolist() for k, v in isaret.items()}


def main():
    klasorler = sorted(p for p in MASKE_KLASORU.iterdir() if p.is_dir())
    if not klasorler:
        raise SystemExit("Maske bulunamadı. Önce 01_veri_indir.py çalıştırılmalı.")
    print(f"{len(klasorler)} denek ölçülüyor...")
    satirlar, isaretler = [], {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as havuz:
        for i, (s, isr) in enumerate(havuz.map(denek_olc, klasorler, chunksize=4), 1):
            satirlar.append(s)
            if isr:
                isaretler[s["denek"]] = isr
            if i % 100 == 0:
                print(f"  {i}/{len(klasorler)}")
    df = pd.DataFrame(satirlar)
    meta = MASKE_KLASORU / "meta.csv"
    if meta.exists():
        m = pd.read_csv(meta, sep=None, engine="python")
        idk = "image_id" if "image_id" in m.columns else m.columns[0]
        df = df.merge(m.rename(columns={idk: "denek"}), on="denek", how="left")
    df.to_csv(CIKTI / "bt_olcumler.csv", index=False)
    (CIKTI / "bt_yer_isaretleri.json").write_text(json.dumps(isaretler))
    sayilar = df.drop(columns=["denek"]).select_dtypes("number").notna().sum()
    print("Geçerli ölçüm sayıları (_aralik_disi: olabilirlik aralığı dışında kalıp çıkarılanlar):\n" + sayilar.to_string())


if __name__ == "__main__":
    main()
