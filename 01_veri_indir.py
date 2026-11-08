"""TotalSegmentator v2.0.1 veri setinden yalnızca gerekli kemik maskelerini ve meta.csv'yi indirir.

Zip dosyasının tamamı (~20+ GB) yerine HTTP range istekleriyle yalnızca gereken dosyalar çekilir.
Sunucu bunu desteklemezse zip'i tarayıcıyla indirip --yerel-zip ile verebilirsin.

Kullanım:
    python 01_veri_indir.py                    # tüm denekler
    python 01_veri_indir.py --max-denek 150    # hızlı deneme
    python 01_veri_indir.py --yerel-zip D:/Totalsegmentator_dataset_v201.zip
"""
import argparse
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ayarlar import KEMIKLER, MASKE_KLASORU, ZENODO_KAYIT

DESEN = re.compile(r"(?:^|/)(s\d{4})/segmentations/([A-Za-z0-9_]+)\.nii\.gz$")


def zenodo_zip_adresi():
    r = requests.get(f"https://zenodo.org/api/records/{ZENODO_KAYIT}", timeout=60)
    r.raise_for_status()
    dosyalar = r.json()["files"]
    zipler = [f for f in dosyalar if f["key"].endswith(".zip")]
    if not zipler:
        sys.exit("Zenodo kaydında zip dosyası bulunamadı.")
    en_buyuk = max(zipler, key=lambda f: f["size"])
    print(f"Zip: {en_buyuk['key']} ({en_buyuk['size'] / 1e9:.1f} GB)")
    return en_buyuk["links"]["self"]


def gerekli_uyeler(adlar, max_denek):
    istenen = set(KEMIKLER)
    secilen, meta = [], None
    for ad in adlar:
        if ad.endswith("meta.csv"):
            meta = ad
            continue
        m = DESEN.search(ad)
        if m and m.group(2) in istenen:
            secilen.append((m.group(1), m.group(2), ad))
    denekler = sorted({d for d, _, _ in secilen})
    if max_denek:
        denekler = set(denekler[:max_denek])
        secilen = [s for s in secilen if s[0] in denekler]
    return meta, secilen


def hedef_yol(denek, kemik):
    return MASKE_KLASORU / denek / f"{kemik}.nii.gz"


def uzaktan_indir(url, max_denek, is_parcacigi):
    from remotezip import RemoteZip

    print("Zip içindekiler okunuyor (birkaç dakika sürebilir)...")
    with RemoteZip(url, timeout=120) as rz:
        adlar = rz.namelist()
        meta, secilen = gerekli_uyeler(adlar, max_denek)
        if meta:
            (MASKE_KLASORU / "meta.csv").write_bytes(rz.read(meta))
    yapilacak = [s for s in secilen if not hedef_yol(s[0], s[1]).exists()]
    print(f"{len({s[0] for s in secilen})} denek, {len(secilen)} maske; indirilecek: {len(yapilacak)}")

    yerel = threading.local()

    def zip_al():
        if not hasattr(yerel, "rz"):
            yerel.rz = RemoteZip(url, timeout=120)
        return yerel.rz

    def indir(oge):
        denek, kemik, ad = oge
        son = None
        for deneme in range(6):
            try:
                veri = zip_al().read(ad)
                yol = hedef_yol(denek, kemik)
                yol.parent.mkdir(parents=True, exist_ok=True)
                gecici = yol.with_suffix(".part")
                gecici.write_bytes(veri)
                gecici.replace(yol)
                return True
            except Exception as hata:  # ağ hatasında yeniden dene
                if hasattr(yerel, "rz"):
                    del yerel.rz
                time.sleep(5 * (deneme + 1))   # sunucu hız sınırı (429) için artan bekleme
                son = hata
        print(f"  ! {ad}: {son}")
        return False

    basla, tamam = time.time(), 0
    with ThreadPoolExecutor(is_parcacigi) as havuz:
        for i, f in enumerate(as_completed([havuz.submit(indir, o) for o in yapilacak]), 1):
            tamam += f.result()
            if i % 200 == 0 or i == len(yapilacak):
                hiz = i / max(time.time() - basla, 1)
                kalan = (len(yapilacak) - i) / max(hiz, 1e-6) / 60
                print(f"  {i}/{len(yapilacak)}  ({hiz:.1f} dosya/sn, ~{kalan:.0f} dk kaldı)")
                if i == 400 and kalan > 120:
                    print("  UYARI: Bu hızla indirme 2 saati aşacak. Ctrl+C ile durdurup zip'i tarayıcıdan indirmeyi\n"
                          "  (python 01_veri_indir.py --yerel-zip <yol>) ya da --max-denek ile daha az denek kullanmayı düşün.")
    print(f"Bitti: {tamam} dosya indirildi -> {MASKE_KLASORU}")


def yerelden_cikar(zip_yolu, max_denek):
    with zipfile.ZipFile(zip_yolu) as z:
        meta, secilen = gerekli_uyeler(z.namelist(), max_denek)
        if meta:
            (MASKE_KLASORU / "meta.csv").write_bytes(z.read(meta))
        for i, (denek, kemik, ad) in enumerate(secilen, 1):
            yol = hedef_yol(denek, kemik)
            if not yol.exists():
                yol.parent.mkdir(parents=True, exist_ok=True)
                yol.write_bytes(z.read(ad))
            if i % 1000 == 0:
                print(f"  {i}/{len(secilen)}")
    print(f"Bitti -> {MASKE_KLASORU}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-denek", type=int, default=0, help="0 = hepsi")
    p.add_argument("--yerel-zip", type=Path)
    p.add_argument("--url", help="Zip adresini elle vermek için")
    p.add_argument("--is-parcacigi", type=int, default=6)
    a = p.parse_args()
    if a.yerel_zip:
        return yerelden_cikar(a.yerel_zip, a.max_denek)
    url = a.url or zenodo_zip_adresi()
    bas = requests.head(url, allow_redirects=True, timeout=60)
    if bas.headers.get("Accept-Ranges", "").lower() != "bytes":
        print("Uyarı: sunucu 'Accept-Ranges: bytes' bildirmedi; yine de deneniyor.")
    try:
        uzaktan_indir(url, a.max_denek, a.is_parcacigi)
    except Exception as hata:
        print(f"\nSeçici indirme başarısız oldu: {hata}")
        print("Çözüm: zip dosyasını tarayıcıyla indir ve şunu çalıştır:")
        print("  python 01_veri_indir.py --yerel-zip <zip dosyasının yolu>")
        print(f"Zenodo sayfası: https://zenodo.org/records/{ZENODO_KAYIT}")
        sys.exit(1)


if __name__ == "__main__":
    main()
