"""BT ölçümlerinden anatomik olabilirlik modelini eğitir ve BT referans raporunu üretir.

Model: kemik uzunluklarının logaritması için çok değişkenli Gauss dağılımı. Her BT tüm vücudu
kapsamadığından eksik veriyle Beklenti-Maksimizasyon (EM) algoritması kullanılır; kovaryans
küçültme (shrinkage) ile kararlı tutulur. Model 5 katlı çapraz doğrulamayla sınanır.

Çıktılar: cikti/bt_model.json, cikti/bt_referans.csv, cikti/bt_referans_raporu.md, cikti/sekil_bt_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from ayarlar import CIKTI, DEGISKENLER

KUCULTME = 0.15
ACIKLAMA = {
    "kafatasi_cap2": "Kafatası ikinci ana çapı",
    "humerus_sol": "Humerus (sol)", "humerus_sag": "Humerus (sağ)",
    "femur_sol": "Femur (sol)", "femur_sag": "Femur (sağ)",
    "juguler_omuz_sol": "Juguler çentik–omuz (sol)", "juguler_omuz_sag": "Juguler çentik–omuz (sağ)",
    "omurga_C7_L5": "Omurga yayı C7–L5", "kalca_genisligi": "İki femur başı arası",
}


def em_gauss(X, iterasyon=200, tol=1e-7):
    n, p = X.shape
    gozlenen = ~np.isnan(X)
    satir = gozlenen.any(1)
    X, gozlenen = X[satir], gozlenen[satir]
    mu = np.nanmean(X, 0)
    S = np.diag(np.nanvar(X, 0) + 1e-6)
    for _ in range(iterasyon):
        toplam, carpim = np.zeros(p), np.zeros((p, p))
        for x, o in zip(X, gozlenen):
            m = ~o
            xh, C = x.copy(), np.zeros((p, p))
            if m.any():
                Soo_inv = np.linalg.inv(S[np.ix_(o, o)])
                K = S[np.ix_(m, o)] @ Soo_inv
                xh[m] = mu[m] + K @ (x[o] - mu[o])
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - K @ S[np.ix_(o, m)]
            toplam += xh
            carpim += np.outer(xh, xh) + C
        yeni_mu = toplam / len(X)
        yeni_S = carpim / len(X) - np.outer(yeni_mu, yeni_mu)
        yeni_S = (1 - KUCULTME) * yeni_S + KUCULTME * np.diag(np.diag(yeni_S))
        fark = np.abs(yeni_mu - mu).max() + np.abs(yeni_S - S).max()
        mu, S = yeni_mu, yeni_S
        if fark < tol:
            break
    return mu, S


def mahalanobis_gozlenen(x, mu, S):
    o = ~np.isnan(x)
    d = x[o] - mu[o]
    return float(d @ np.linalg.inv(S[np.ix_(o, o)]) @ d), int(o.sum())


def main():
    df = pd.read_csv(CIKTI / "bt_olcumler.csv")
    for d in DEGISKENLER:
        if d not in df:
            df[d] = np.nan
    X = np.log(df[DEGISKENLER].to_numpy(float))
    n_gecerli = (~np.isnan(X)).sum(0)
    print("Değişken başına geçerli BT sayısı:", dict(zip(DEGISKENLER, n_gecerli.tolist())))
    eksik = [d for d, n in zip(DEGISKENLER, n_gecerli) if n < 5]
    if eksik:
        print(f"UYARI: Şu değişkenler için yeterli BT yok (<5), modelden çıkarıldı: {eksik}")
    if "kafatasi_cap2" in eksik:
        raise SystemExit("Kafatası ölçümü olan BT yok; ölçek çapası olmadan model kurulamaz. Daha fazla denek indirin.")
    secili = [d for d in DEGISKENLER if d not in eksik]
    X = X[:, [DEGISKENLER.index(d) for d in secili]]
    n_gecerli = (~np.isnan(X)).sum(0)

    mu, S = em_gauss(X)

    # 5 katlı çapraz doğrulama: dışarıda bırakılan BT'lerin Mahalanobis uzaklıkları ki-kare ile uyumlu mu?
    rng = np.random.default_rng(42)
    kat = rng.integers(0, 5, len(X))
    p_degerleri = []
    for f in range(5):
        m_, S_ = em_gauss(X[kat != f])
        for x in X[kat == f]:
            if (~np.isnan(x)).sum() >= 2:
                d2, k = mahalanobis_gozlenen(x, m_, S_)
                p_degerleri.append(chi2.sf(d2, k))
    p_degerleri = np.array(p_degerleri)
    yanlis_alarm = float((p_degerleri < 0.05).mean()) if len(p_degerleri) else float("nan")

    model = {"degiskenler": secili, "mu": mu.tolist(), "sigma": S.tolist(),
             "n_gecerli": dict(zip(secili, n_gecerli.tolist())), "n_bt": int(len(df)),
             "kucultme": KUCULTME, "cv_yanlis_alarm_orani": yanlis_alarm, "cv_n": int(len(p_degerleri))}
    (CIKTI / "bt_model.json").write_text(json.dumps(model, indent=2))

    # Referans tablosu (mm)
    satirlar = []
    for d in DEGISKENLER:
        v = df[d].dropna()
        satirlar.append({"olcut": ACIKLAMA[d], "degisken": d, "n": len(v), "ortalama_mm": v.mean(),
                         "ss_mm": v.std(), "p2.5_mm": v.quantile(0.025), "medyan_mm": v.median(),
                         "p97.5_mm": v.quantile(0.975)})
    ref = pd.DataFrame(satirlar)
    ref.to_csv(CIKTI / "bt_referans.csv", index=False)

    cins = next((c for c in ("gender", "sex", "cinsiyet") if c in df.columns), None)
    yas = next((c for c in ("age", "yas") if c in df.columns), None)

    # Şekiller
    fig, eksenler = plt.subplots(3, 3, figsize=(11, 9))
    for ax, d in zip(eksenler.flat, DEGISKENLER):
        v = df[d].dropna()
        ax.hist(v, bins=30, color="#0e7c86", alpha=0.85)
        ax.set_title(f"{ACIKLAMA[d]} (n={len(v)})", fontsize=9)
        ax.set_xlabel("mm", fontsize=8)
    fig.suptitle("BT referans dağılımları (TotalSegmentator v2.0.1)")
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_bt_dagilimlar.png", dpi=150)
    plt.close(fig)

    kor = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(kor, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(secili)), [ACIKLAMA[d] for d in secili], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(secili)), [ACIKLAMA[d] for d in secili], fontsize=7)
    fig.colorbar(im, label="korelasyon (log uzunluk)")
    ax.set_title("Modelin öğrendiği kemik uzunluğu korelasyonları")
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_bt_korelasyon.png", dpi=150)
    plt.close(fig)

    # Rapor
    r = ["# BT Referans Raporu", "",
         f"Kaynak: TotalSegmentator v2.0.1 (Wasserthal vd., 2023), CC BY 4.0. İşlenen BT sayısı: {len(df)}.",
         "Görüntü sınırına değen (kesik) kemikler ve BT'nin dairesel görüntü alanı nedeniyle eksik kalıp erişkin "
         "olabilirlik aralığının dışına düşen parçalar ölçüme alınmamıştır.", ""]
    disari = {c.replace("_aralik_disi", ""): int(df[c].notna().sum()) for c in df.columns if c.endswith("_aralik_disi")}
    if disari:
        r.append("Aralık dışı kalıp çıkarılan ölçüm sayıları: " + ", ".join(f"{k}: {v}" for k, v in disari.items()))
        r.append("")
    if cins:
        r.append(f"Cinsiyet dağılımı: {df[cins].value_counts().to_dict()}")
    if yas:
        r.append(f"Yaş: ortalama {df[yas].mean():.1f}, aralık {df[yas].min():.0f}–{df[yas].max():.0f}")
    r += ["", "## Kemik uzunlukları (mm)", "",
          "| Ölçüt | n | Ortalama | SS | %2,5 | Medyan | %97,5 |", "|---|---|---|---|---|---|---|"]
    for _, s in ref.iterrows():
        r.append(f"| {s.olcut} | {s.n} | {s.ortalama_mm:.1f} | {s.ss_mm:.1f} | {s['p2.5_mm']:.1f} | "
                 f"{s.medyan_mm:.1f} | {s['p97.5_mm']:.1f} |")
    r += ["", "## Model", "",
          f"Çok değişkenli Gauss (log uzunluk), eksik veri için EM, küçültme katsayısı {KUCULTME}.",
          f"5 katlı çapraz doğrulamada dışarıda bırakılan {len(p_degerleri)} BT'nin "
          f"%{100 * yanlis_alarm:.1f}'i yanlışlıkla 'olanaksız' (p<0,05) bulunmuştur (beklenen ≈ %5).",
          "", "![Dağılımlar](sekil_bt_dagilimlar.png)", "", "![Korelasyon](sekil_bt_korelasyon.png)"]
    (CIKTI / "bt_referans_raporu.md").write_text("\n".join(r), encoding="utf-8")
    print(ref.round(1).to_string(index=False))
    print(f"Çapraz doğrulama yanlış alarm oranı: {yanlis_alarm:.3f} (n={len(p_degerleri)})")
    print("Model kaydedildi -> cikti/bt_model.json")


if __name__ == "__main__":
    main()
