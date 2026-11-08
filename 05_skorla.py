"""İşaretlenen görselleri BT modeliyle skorlar, kaynakları karşılaştırır ve rapor üretir.

Dosya adı kuralı:
    ogrenci_P1.png               öğrencinin çizimi (P1–P6)
    MODEL_KOSUL_POZ_rN.png       yapay zekâ görseli, ör. gemini_A_P3_r2.png

Çıktılar: cikti/skorlar.csv, cikti/ozet.csv, cikti/rapor.html, cikti/sekil_*.png
"""
import base64
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from ayarlar import CIKTI, EK_SEGMENTLER, ISARETLER, SEGMENTLER
from skor_cekirdek import model_yukle, skorla, skorla_mc

AD = re.compile(r"^(?P<kaynak>[A-Za-z0-9\-]+?)_(?:(?P<kosul>[AB])_)?(?P<poz>P\d)(?:_r(?P<tekrar>\d+))?\.", re.I)
SEG_AD = {"humerus_a": "Humerus", "humerus_b": "Humerus", "femur_a": "Femur", "femur_b": "Femur",
          "juguler_omuz_a": "Juguler–omuz", "juguler_omuz_b": "Juguler–omuz",
          "omurga_C7_L5": "Omurga C7–L5", "kalca_genisligi": "Kalça genişliği"}


def isaretleri_oku(kaynak):
    """kaynak: 'otomatik' (yalnızca model), 'insan' (elle işaretlenenler) veya 'hepsi'."""
    kayit = {}
    for yol in sorted(ISARETLER.glob("*.json")):
        oto = yol.name.startswith("otomatik")
        if (kaynak == "otomatik" and not oto) or (kaynak == "insan" and oto):
            continue
        v = json.loads(yol.read_text(encoding="utf-8"))
        for g in v.get("gorseller", []):
            kayit[g["dosya"]] = g          # sonraki dosya öncekinin üzerine yazar
    return list(kayit.values())


def uzaklik(n, a, b):
    if a in n and b in n:
        return float(np.hypot(n[a][0] - n[b][0], n[a][1] - n[b][1]))
    return None


def satir_uret(model, g):
    m = AD.match(g["dosya"])
    if not m:
        print(f"  ! Dosya adı kurala uymuyor, atlandı: {g['dosya']}")
        return None
    n = g.get("noktalar", {})
    s = {"dosya": g["dosya"], "kaynak": m["kaynak"].lower(), "kosul": m["kosul"] or "-", "poz": m["poz"].upper(),
         "tekrar": int(m["tekrar"]) if m["tekrar"] else 0, "talimat_disi": bool(g.get("talimat_disi"))}
    uz = {seg: uzaklik(n, a, b) for seg, (a, b, _) in SEGMENTLER.items()}
    uz = {k: v for k, v in uz.items() if v}
    kafa = uzaklik(n, "kafatasi_1", "kafatasi_2")
    sonuc = skorla(model, uz, kafa) if not s["talimat_disi"] else None
    if sonuc:
        s.update(d2=sonuc["d2"], olasilik=sonuc["olasilik"], olanaksiz=sonuc["olanaksiz"],
                 asim_sayisi=sonuc["asim_sayisi"], olculen_segment=sonuc["k"])
        for seg, d in sonuc["segmentler"].items():
            s[f"oran_{seg}"] = d["medyana_gore"]
            s[f"asim_{seg}"] = d["asim"]
    mc = skorla_mc(model, uz, kafa) if not s["talimat_disi"] else None
    if mc:
        s.update(mc_p=mc["mc_p"], mc_d2=mc["mc_d2"], mc_gercek_disi=mc["mc_gercek_disi"], mc_k=mc["mc_k"])
        for seg, d in mc["mc_segmentler"].items():
            s[f"z_{seg}"] = d["z"]
        # Duyarlılık analizi: kemik yönleri tamamen rastgele (en temkinli; güçlü kısalmaya izin verir)
        mc_r = skorla_mc(model, uz, kafa, gorus="rastgele")
        if mc_r:
            s.update(mc_p_rastgele=mc_r["mc_p"], mc_gercek_disi_rastgele=mc_r["mc_gercek_disi"])
    for seg, (a, b) in EK_SEGMENTLER.items():
        ust = "humerus_" + seg[-1] if seg.startswith("onkol") else "femur_" + seg[-1]
        l, u = uzaklik(n, a, b), uz.get(ust)
        if l and u:
            s[f"ek_{seg}_orani"] = l / u
    for alan, beklenen in (("kaburga_a", 12), ("kaburga_b", 12), ("lomber", 5), ("parmak_el", 5)):
        if g.get(alan) is not None:
            s[alan] = g[alan]
            s[f"{alan}_sapma"] = abs(g[alan] - beklenen)
    return s


def resim64(yol):
    return base64.b64encode(yol.read_bytes()).decode()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--isaret", choices=["otomatik", "insan", "hepsi"], default=None,
                   help="Varsayılan: isaretler/otomatik.json varsa otomatik, yoksa insan")
    a = p.parse_args()
    kaynak = a.isaret or ("otomatik" if (ISARETLER / "otomatik.json").exists() else "insan")
    print("İşaret kaynağı:", kaynak)
    model = model_yukle()
    kayitlar = isaretleri_oku(kaynak)
    if not kayitlar:
        raise SystemExit("isaretler klasöründe JSON yok. Önce isaretleme_araci.html ile işaretleyin.")
    df = pd.DataFrame([r for r in (satir_uret(model, g) for g in kayitlar) if r])
    df.to_csv(CIKTI / "skorlar.csv", index=False)
    gecerli = df[~df.talimat_disi & df.get("mc_d2", pd.Series(dtype=float)).notna()].copy()
    gecerli["d2"] = gecerli["mc_d2"]   # birincil skor: iki yönlü Monte Carlo

    sira = ["ogrenci"] + sorted(k for k in df.kaynak.unique() if k != "ogrenci")
    sira = [k for k in sira if k in df.kaynak.unique()]

    def ozet_al(g):
        d = {"görsel": len(g), "talimat dışı %": 100 * g.talimat_disi.mean()}
        v = g[~g.talimat_disi]
        if "mc_p" in v and v.mc_p.notna().any():
            d.update({"gerçek dışı % (MC)": 100 * v.mc_gercek_disi.astype(float).mean(),
                      "medyan p (MC)": v.mc_p.median(), "medyan d² (MC)": v.mc_d2.median()})
            if "mc_gercek_disi_rastgele" in v:
                d["gerçek dışı % (rastgele açı)"] = 100 * v.mc_gercek_disi_rastgele.astype(float).mean()
        if "d2" in v and v.d2.notna().any():
            d.update({"olanaksız % (üst sınır)": 100 * v.olanaksiz.astype(float).mean(),
                      "ort. oran aşımı": v.asim_sayisi.mean()})
        for alan in ("kaburga_a_sapma", "kaburga_b_sapma", "lomber_sapma", "parmak_el_sapma"):
            if alan in v and v[alan].notna().any():
                d[alan.replace("_sapma", " sapma (ort.)")] = v[alan].mean()
        return pd.Series(d)

    ozet = df.groupby(["kaynak", "kosul"]).apply(ozet_al, include_groups=False).reset_index()
    ozet.to_csv(CIKTI / "ozet.csv", index=False)

    # İstatistik
    istat = []
    gruplar = [gecerli[gecerli.kaynak == k].d2.values for k in sira if (gecerli.kaynak == k).sum() >= 2]
    if len(gruplar) >= 2:
        H, p = kruskal(*gruplar)
        istat.append(f"Kruskal-Wallis (tüm kaynaklar, d²): H = {H:.2f}, p = {p:.4f}")
    ogr = gecerli[gecerli.kaynak == "ogrenci"].d2.values
    karsilastirma = []
    for k in sira[1:]:
        yz = gecerli[gecerli.kaynak == k].d2.values
        if len(ogr) >= 2 and len(yz) >= 2:
            U, p = mannwhitneyu(ogr, yz, alternative="two-sided")
            karsilastirma.append((k, U, p))
    m = len(karsilastirma)
    for i, (k, U, p) in enumerate(sorted(karsilastirma, key=lambda x: x[2])):
        p_holm = min(1.0, p * (m - i))
        istat.append(f"Öğrenci – {k}: Mann-Whitney U = {U:.1f}, p = {p:.4f}, Holm düzeltilmiş p = {p_holm:.4f}")

    # Şekil 1: d² dağılımı
    fig, ax = plt.subplots(figsize=(7, 4))
    veriler = [gecerli[gecerli.kaynak == k].d2.values for k in sira]
    ax.boxplot([v if len(v) else [np.nan] for v in veriler], showfliers=False)
    for i, v in enumerate(veriler, 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(v)), v, s=14, alpha=0.7,
                   color="#e0115f" if sira[i - 1] == "ogrenci" else "#0e7c86")
    ax.set_xticks(range(1, len(sira) + 1), sira)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("BT tabanlı sapma skoru d² (küçük = gerçekçi)")
    ax.set_title("Kaynağa göre BT tabanlı gerçekçilik skoru")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_skor_kaynak.png", dpi=150)
    plt.close(fig)

    # Şekil 2: segment bazında BT %97,5 sınırını aşma oranı
    asim_sut = [c for c in gecerli.columns if c.startswith("asim_") and c != "asim_sayisi"]
    if asim_sut:
        tip = {}
        for c in asim_sut:
            tip.setdefault(SEG_AD[c[5:]], []).append(c)
        tablo = pd.DataFrame({ad: gecerli.groupby("kaynak")[cs].apply(
            lambda x: np.nanmean(x.to_numpy(dtype=float))) for ad, cs in tip.items()}).reindex(sira)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        im = ax.imshow(100 * tablo.values.astype(float), cmap="Reds", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(tablo.shape[1]), tablo.columns, rotation=20)
        ax.set_yticks(range(tablo.shape[0]), tablo.index)
        for (i, j), v in np.ndenumerate(tablo.values.astype(float)):
            if not np.isnan(v):
                ax.text(j, i, f"%{100 * v:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, label="BT %97,5 sınırını aşan görsel (%)")
        ax.set_title("Hangi kemik gerçekte olabileceğinden uzun çizildi?")
        fig.tight_layout()
        fig.savefig(CIKTI / "sekil_segment_asim.png", dpi=150)
        plt.close(fig)

    # Şekil 3: poz bazında öğrenci ve yapay zekâ
    fig, ax = plt.subplots(figsize=(7, 3.6))
    pozlar = sorted(gecerli.poz.unique())
    for k in sira:
        g = gecerli[gecerli.kaynak == k].groupby("poz").d2.median().reindex(pozlar)
        ax.plot(pozlar, g.values, marker="o", lw=2.5 if k == "ogrenci" else 1.2, label=k)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Medyan d²")
    ax.set_title("Pozlara göre skor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_poz.png", dpi=150)
    plt.close(fig)

    bt = model_yukle()
    basliklar = {"sekil_skor_kaynak.png": "Kaynağa göre skor", "sekil_segment_asim.png": "Kemik bazında BT sınırı aşımı",
                 "sekil_poz.png": "Pozlara göre skor", "sekil_dogrulama.png": "Sistemin BT ile doğrulanması",
                 "sekil_bt_dagilimlar.png": "BT referans dağılımları"}
    sekiller = [p for p in basliklar if (CIKTI / p).exists()]
    html = ["<!doctype html><meta charset='utf-8'><title>İskelet Gerçekçilik Raporu</title>",
            "<style>body{font:15px/1.5 system-ui,'Segoe UI',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#17212b}"
            "table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right}"
            "th:first-child,td:first-child{text-align:left}img{max-width:100%}code{background:#eef2f4;padding:1px 4px}</style>",
            "<h1>İskelet Görsellerinin BT Tabanlı Anatomik Gerçekçilik Raporu</h1>",
            f"<p>BT modeli: {bt['n_bt']} BT (TotalSegmentator v2.0.1). Çapraz doğrulama yanlış alarm oranı: "
            f"%{100 * bt['cv_yanlis_alarm_orani']:.1f}. Değerlendirilen görsel: {len(df)} "
            f"(talimat dışı: {int(df.talimat_disi.sum())}). Yer işaretleri: <b>{kaynak}</b>.</p>",
            "<p><b>Birincil skor (MC)</b>: BT modelinden örneklenen 40.000 gerçek bedenin çizim benzeri açılardan "
            "izdüşümleriyle karşılaştırma. <b>d²</b> görselin kemik/kafatası oranlarının bu dağılımdan sapmasıdır; "
            "<b>p</b> gerçek bedenlerin çizimlerinde bu kadar sıra dışı oranların görülme olasılığıdır. "
            "p &lt; 0,05 ise görsel <b>gerçek dışı</b> sayılır. İkincil skor (üst sınır testi) yalnızca hiçbir açıdan "
            "olanaksız kemik uzunluklarını yakalar.</p>",
            "<h2>Özet</h2>", ozet.round(2).to_html(index=False), "<h2>İstatistik</h2>",
            "<ul>" + "".join(f"<li>{s}</li>" for s in istat) + "</ul>" if istat else "<p>Karşılaştırma için yeterli görsel yok.</p>"]
    for p in sekiller:
        html.append(f"<h3>{basliklar[p]}</h3><img src='data:image/png;base64,{resim64(CIKTI / p)}'>")
    html.append("<h2>Görsel bazında skorlar</h2>")
    goster = [c for c in ["dosya", "kaynak", "kosul", "poz", "talimat_disi", "mc_p", "mc_d2", "mc_gercek_disi", "d2", "olanaksiz",
                          "asim_sayisi", "kaburga_a", "kaburga_b", "lomber", "parmak_el"] if c in df]
    html.append(df[goster].round(3).to_html(index=False))
    (CIKTI / "rapor.html").write_text("\n".join(html), encoding="utf-8")
    print(ozet.round(2).to_string(index=False))
    print("\n".join(istat))
    print("Rapor -> cikti/rapor.html")


if __name__ == "__main__":
    main()
