"""Skorlama sisteminin BT verisiyle doğrulanması (simülasyon).

1) Özgüllük: Gerçek BT bedenleri rastgele açılardan ve rastgele kol/bacak pozlarında 2B'ye izdüşürülür.
   Bunlar gerçek insanlar olduğundan sistem "olanaksız" dememelidir.
2) Duyarlılık: Aynı izdüşümlerde bir kemik %15, %30 veya %50 uzatılır; sistem bunu yakalamalıdır.

Kafatası 3B'de bir elipsoit olarak alınır; 2B'deki en uzun çapı izdüşümden hesaplanır.
BT'de kafatası ve en az bir başka kemik birlikte ölçülmüş denekler (gerçek) ile modelden örneklenen
bedenler (sentetik) ayrı ayrı raporlanır.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ayarlar import CIKTI, DEGISKENLER
from skor_cekirdek import model_yukle, skorla, skorla_mc

TEKRAR = 20
CARPANLAR = [1.0, 1.15, 1.3, 1.5]
rng = np.random.default_rng(7)

SEG_DEG = {"humerus_a": "humerus_sag", "humerus_b": "humerus_sol", "femur_a": "femur_sag",
           "femur_b": "femur_sol", "juguler_omuz_a": "juguler_omuz_sag", "juguler_omuz_b": "juguler_omuz_sol",
           "omurga_C7_L5": "omurga_C7_L5", "kalca_genisligi": "kalca_genisligi"}
GOVDE_VEKTOR = {"juguler_omuz_sag": ("juguler", "omuz_sag"), "juguler_omuz_sol": ("juguler", "omuz_sol"),
                "omurga_C7_L5": ("C7", "L5"), "kalca_genisligi": ("kalca_sol", "kalca_sag")}


def rastgele_yon():
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def izdusur(uzunluklar, isaretler, caplar):
    """Bir bedeni rastgele açıdan 2B'ye izdüşürür. Uzuvlar rastgele yönlerde, gövde katı döner."""
    R = Rotation.random(random_state=rng).as_matrix()
    iki_b = {}
    for seg, d in SEG_DEG.items():
        L = uzunluklar.get(d)
        if L is None or np.isnan(L):
            continue
        if d in GOVDE_VEKTOR and isaretler and all(k in isaretler for k in GOVDE_VEKTOR[d]):
            v = np.subtract(isaretler[GOVDE_VEKTOR[d][1]], isaretler[GOVDE_VEKTOR[d][0]])
        else:
            v = rastgele_yon() * L
            if d == "omurga_C7_L5":
                v *= 0.95      # yay yerine kiriş: kiriş yaydan kısadır
        iki_b[seg] = float(np.linalg.norm((R @ v)[:2]))
    a, b, c = caplar
    M = (R @ np.diag([a, b, c]))[:2]
    kafa_2b = float(np.linalg.svd(M, compute_uv=False)[0])
    return iki_b, kafa_2b


def izdusur_cizim(uzunluklar, caplar):
    """Çizim benzeri izdüşüm: her kemik görüntü düzlemine en fazla 45° eğik; kafatası rastgele döner.
    Kafatası çizimdeki elipsoit geometrisiyle (g/2, 0,95g/2, 1,25g/2) üretilir."""
    iki_b = {}
    for seg, d in SEG_DEG.items():
        L = uzunluklar.get(d)
        if L is None or np.isnan(L):
            continue
        iki_b[seg] = float(L * np.cos(np.radians(rng.uniform(-45, 45))))
    g = caplar[1]
    R = Rotation.random(random_state=rng).as_matrix()
    M = (R @ np.diag([g / 2, 0.95 * g / 2, 1.25 * g / 2]))[:2]
    return iki_b, float(2 * np.linalg.svd(M, compute_uv=False)[0])


def calistir_mc(bedenler, etiket, model, carpanlar=(0.7, 1.0, 1.15, 1.3, 1.5)):
    satirlar = []
    for uz, _, cap in bedenler:
        for _ in range(TEKRAR):
            iki_b, kafa = izdusur_cizim(uz, cap)
            if not iki_b:
                continue
            hedef = rng.choice(list(iki_b))
            for f in carpanlar:
                kopya = dict(iki_b)
                kopya[hedef] *= f
                s = skorla_mc(model, kopya, kafa)
                if s:
                    satirlar.append({"kume": etiket, "carpan": f, "gercek_disi": s["mc_gercek_disi"], "p": s["mc_p"]})
    return satirlar


def calistir(bedenler, etiket, model):
    satirlar = []
    for uz, isr, cap in bedenler:
        for _ in range(TEKRAR):
            iki_b, kafa = izdusur(uz, isr, cap)
            if not iki_b:
                continue
            hedef = rng.choice(list(iki_b))
            for f in CARPANLAR:
                kopya = dict(iki_b)
                kopya[hedef] *= f
                s = skorla(model, kopya, kafa)
                if s:
                    satirlar.append({"kume": etiket, "carpan": f, "olanaksiz": s["olanaksiz"],
                                     "asim_var": s["asim_sayisi"] > 0, "d2": s["d2"], "k": s["k"]})
    return satirlar


def main():
    model = model_yukle()
    df = pd.read_csv(CIKTI / "bt_olcumler.csv")
    isaretler = json.loads((CIKTI / "bt_yer_isaretleri.json").read_text())

    oran1 = (df["kafatasi_cap1"] / df["kafatasi_cap2"]).median()
    oran3 = (df["kafatasi_cap3"] / df["kafatasi_cap2"]).median()

    gercek = []
    for _, s in df.iterrows():
        if pd.isna(s.get("kafatasi_cap2")):
            continue
        uz = {d: s.get(d) for d in DEGISKENLER if d != "kafatasi_cap2"}
        if sum(pd.notna(v) for v in uz.values()) >= 1:
            gercek.append((uz, isaretler.get(s["denek"]), (s.kafatasi_cap1, s.kafatasi_cap2, s.kafatasi_cap3)))

    sentetik = []
    for x in rng.multivariate_normal(model["mu"], model["sigma"], size=200):
        L = dict(zip(model["degiskenler"], np.exp(x)))
        c2 = L.pop("kafatasi_cap2")
        sentetik.append((L, None, (c2 * oran1, c2, c2 * oran3)))

    satirlar = calistir(gercek, f"gerçek BT (n={len(gercek)})", model) + \
        calistir(sentetik, "modelden örneklenen (n=200)", model)
    # İki yönlü Monte Carlo skoru: modelden BAĞIMSIZ örneklenen bedenler ve gerçek BT bedenleri
    bagimsiz = []
    for x in np.random.default_rng(99).multivariate_normal(model["mu"], model["sigma"], size=200):
        L = dict(zip(model["degiskenler"], np.exp(x)))
        c2 = L.pop("kafatasi_cap2")
        bagimsiz.append((L, None, (None, c2, None)))
    mc = pd.DataFrame(calistir_mc(gercek, f"gerçek BT (n={len(gercek)})", model) +
                      calistir_mc(bagimsiz, "modelden örneklenen (n=200)", model))
    mc.to_csv(CIKTI / "dogrulama_mc.csv", index=False)
    mc_tablo = mc.groupby(["kume", "carpan"]).agg(oran=("gercek_disi", "mean"), n=("p", "size")).reset_index()
    sonuc = pd.DataFrame(satirlar)
    sonuc.to_csv(CIKTI / "dogrulama.csv", index=False)
    tablo = sonuc.groupby(["kume", "carpan"]).agg(olanaksiz_orani=("olanaksiz", "mean"),
                                                  asim_orani=("asim_var", "mean"), n=("d2", "size")).reset_index()

    r = ["# Skorlama Sisteminin Doğrulanması", "",
         "Çarpan 1,0 satırları gerçek bedenlerdir: buradaki 'olanaksız' oranı yanlış alarm oranıdır (düşük olmalı).",
         "Çarpan > 1 satırlarında rastgele bir kemik uzatılmıştır: buradaki oran yakalama gücüdür (yüksek olmalı).",
         "Perspektif kısalması nedeniyle uzatılan kemik görüntüde zaten kısa görünüyorsa hata yakalanamaz; "
         "bu yöntemin doğal sınırıdır.", "",
         "| Küme | Kemik çarpanı | n | 'Olanaksız' oranı | En az bir oran aşımı |", "|---|---|---|---|---|"]
    for _, s in tablo.iterrows():
        r.append(f"| {s.kume} | {s.carpan:.2f} | {s.n} | %{100 * s.olanaksiz_orani:.1f} | %{100 * s.asim_orani:.1f} |")
    r += ["", "## İki yönlü Monte Carlo gerçekçilik skoru (çizim benzeri açılar)", "",
          "Kemikler görüntü düzlemine en fazla 45° eğik varsayılır. Çarpan 1,0 = hatasız (yanlış alarm, ≈%5 beklenir); "
          "0,7 = kemik %30 kısa; 1,15–1,5 = kemik uzun.", "",
          "| Küme | Kemik çarpanı | n | 'Gerçek dışı' (p<0,05) oranı |", "|---|---|---|---|"]
    for _, s in mc_tablo.iterrows():
        r.append(f"| {s.kume} | {s.carpan:.2f} | {s.n} | %{100 * s.oran:.1f} |")
    r += ["", "![Doğrulama](sekil_dogrulama.png)"]
    (CIKTI / "dogrulama_raporu.md").write_text("\n".join(r), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    for kume, g in tablo.groupby("kume"):
        ax.plot(g.carpan, 100 * g.olanaksiz_orani, marker="o", label=kume)
    ax.set_xlabel("Uzatılan kemiğin çarpanı (1,0 = hatasız)")
    ax.set_ylabel("'Olanaksız' bulunma oranı (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Skorlama sisteminin BT ile doğrulanması")
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_dogrulama.png", dpi=150)
    print(tablo.to_string(index=False))
    print(mc_tablo.to_string(index=False))


if __name__ == "__main__":
    main()
