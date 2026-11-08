"""(İsteğe bağlı) Yapay zekâ görsellerini API ile protokole uygun şekilde üretir.

API anahtarı gerekir. Web arayüzlerinden elle üretmek de geçerlidir; o durumda dosyaları
gorseller/yz klasörüne MODEL_KOSUL_POZ_rN.png adıyla koy ve kayit.csv'yi elle doldur.

Kullanım (PowerShell):
    $env:OPENAI_API_KEY="..."; python 06_gorsel_uret_api.py --saglayici openai --model gpt-image-1
    $env:GEMINI_API_KEY="..."; python 06_gorsel_uret_api.py --saglayici gemini --model gemini-2.5-flash-image
Model adları değişebilir; kullandığın tam sürüm adını --model ile ver, kayda o yazılır.
"""
import argparse
import base64
import csv
import os
import time
from datetime import datetime

import requests

from ayarlar import GORSELLER

POZLAR = {
    "P1": "standing on the toes of one foot, with the other knee raised forward and that foot drawn up beside the standing knee; one arm reaches straight up above the head and the other arm extends forward at shoulder height; seen from the side, with the spine slightly arched back",
    "P2": "standing on one straight leg with the torso bent far forward and sideways so that the skull hangs low at hip height, facing the viewer; one arm reaches diagonally up and out; the other leg is bent at the knee with the foot raised behind the body and held by the other hand",
    "P3": "seen from the front, standing on the toes of one foot; the other knee is raised high and turned outward with the foot resting against the standing knee; one hand rests on the hip with the elbow bent outward and the other arm is raised straight above the head; the skull is turned in three-quarter view",
    "P4": "seen from the side, standing on the toes of one straight leg; the other leg is kicked forward and up to hip height with the knee straight; one arm is raised straight up above the head and the other arm reaches forward along the raised leg toward the foot",
    "P5": "in a deep forward bend with both feet on the ground and the legs straight and apart; the torso hangs down so that the skull is at knee height, facing the viewer; both arms are raised up and back above the hips, one straight and one bent at the elbow",
    "P6": "seen in front three-quarter view, standing with the legs crossed, one foot in front of the other and both heels raised; both hands are placed on top of the head with the elbows pointing outward; the torso is slightly twisted and the head is tilted to one side",
}
TEMEL = ("A graphite pencil drawing on plain white paper of one complete adult human skeleton {poz}. The skeleton "
         "should be anatomically accurate, as in a medical anatomy atlas. Show the entire skeleton from the top of "
         "the skull to the tips of the toes, with nothing cropped. Bones only: no skin, no muscles, no clothing. "
         "Black and grey pencil lines with light shading, in the style of an art-school anatomy study. Plain white "
         "background with no other objects and no text. One figure only, centered, portrait orientation.")
AYRINTI = (" Anatomical requirements: 12 pairs of ribs; 7 cervical, 12 thoracic and 5 lumbar vertebrae; a sacrum and "
           "coccyx; a separate radius and ulna in each forearm and a separate tibia and fibula in each lower leg; a "
           "patella on each knee; complete hands, each with 8 carpal bones, 5 metacarpals and 14 phalanges; complete "
           "feet, each with 7 tarsal bones, 5 metatarsals and 14 phalanges; realistic adult proportions.")


def istem(poz, kosul):
    return TEMEL.format(poz=POZLAR[poz]) + (AYRINTI if kosul == "B" else "")


def openai_uret(model, metin):
    r = requests.post("https://api.openai.com/v1/images/generations", timeout=300,
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                      json={"model": model, "prompt": metin, "size": "1024x1536", "n": 1})
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


def gemini_uret(model, metin):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    basliklar = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    govde = {"contents": [{"parts": [{"text": metin}]}],
             "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "2:3"}}}
    r = requests.post(url, headers=basliklar, json=govde, timeout=300)
    if r.status_code == 400:   # bazı sürümler imageConfig kabul etmez
        govde["generationConfig"].pop("imageConfig")
        r = requests.post(url, headers=basliklar, json=govde, timeout=300)
    r.raise_for_status()
    for parca in r.json()["candidates"][0]["content"]["parts"]:
        veri = parca.get("inlineData") or parca.get("inline_data")
        if veri:
            return base64.b64decode(veri["data"])
    raise RuntimeError("Yanıtta görsel yok (model reddetmiş olabilir).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saglayici", choices=["openai", "gemini"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--kaynak-adi", help="Dosya adındaki kısa ad (varsayılan: chatgpt / gemini)")
    p.add_argument("--kosullar", default="A,B")
    p.add_argument("--tekrar", type=int, default=5)
    a = p.parse_args()
    kisa = a.kaynak_adi or {"openai": "chatgpt", "gemini": "gemini"}[a.saglayici]
    uret = openai_uret if a.saglayici == "openai" else gemini_uret
    klasor = GORSELLER / "yz"
    klasor.mkdir(parents=True, exist_ok=True)
    kayit = GORSELLER / "kayit.csv"
    yeni = not kayit.exists()
    with open(kayit, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if yeni:
            w.writerow(["dosya_adi", "model", "surum", "arayuz", "tarih_saat", "kosul", "poz", "tekrar", "durum"])
        for kosul in a.kosullar.split(","):
            for poz in POZLAR:
                for r in range(1, a.tekrar + 1):
                    ad = f"{kisa}_{kosul}_{poz}_r{r}.png"
                    if (klasor / ad).exists():
                        continue
                    zaman = datetime.now().isoformat(timespec="seconds")
                    try:
                        (klasor / ad).write_bytes(uret(a.model, istem(poz, kosul)))
                        durum = "uretildi"
                    except Exception as hata:
                        durum = f"hata: {hata}"[:200]
                    w.writerow([ad, kisa, a.model, f"{a.saglayici} API", zaman, kosul, poz, r, durum])
                    f.flush()
                    print(ad, durum)
                    time.sleep(1)


if __name__ == "__main__":
    main()
