"""Ana sonuçların açı varsayımına duyarlılığı: en büyük kemik eğimi 30°, 45°, 60°, 75° ve tamamen rastgele yön."""
import json, importlib.util
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from skor_cekirdek import model_yukle, skorla_mc
from ayarlar import ISARETLER, CIKTI, SEGMENTLER
s = importlib.util.spec_from_file_location("sk", "05_skorla.py"); sk = importlib.util.module_from_spec(s); s.loader.exec_module(sk)
model = model_yukle()
G = sk.isaretleri_oku("insan")
satir = []
for g in G:
    m = sk.AD.match(g["dosya"])
    if not m or g.get("talimat_disi"):
        continue
    n = g["noktalar"]
    uz = {seg: sk.uzaklik(n, a, b) for seg, (a, b, _) in SEGMENTLER.items()}
    uz = {k: v for k, v in uz.items() if v}
    kafa = sk.uzaklik(n, "kafatasi_1", "kafatasi_2")
    for gorus in (30, 45, 60, 75, "rastgele"):
        r = skorla_mc(model, uz, kafa, gorus=gorus)
        satir.append({"dosya": g["dosya"], "kaynak": m["kaynak"].lower(), "gorus": str(gorus), "d2": r["mc_d2"], "gd": r["mc_gercek_disi"]})
d = pd.DataFrame(satir)
out = []
for gorus, v in d.groupby("gorus", sort=False):
    o, gm, cg = (v[v.kaynak == k].d2 for k in ("ogrenci", "gemini", "chatgpt"))
    pg = mannwhitneyu(o, gm).pvalue; pc = mannwhitneyu(o, cg).pvalue
    ph = sorted([pg, pc]); holm_g = min(1, 2 * pg) if pg <= pc else max(min(1, 2 * pc), pg)
    out.append({"en_buyuk_egim": gorus, "medyan_d2_ogrenci": o.median(), "medyan_d2_gemini": gm.median(), "medyan_d2_chatgpt": cg.median(),
                "ogr_gemini_p_holm": holm_g, "gercek_disi_ogrenci": int(v[v.kaynak == "ogrenci"].gd.sum()),
                "gercek_disi_yz": int(v[v.kaynak != "ogrenci"].gd.sum())})
t = pd.DataFrame(out).round(3)
t.to_csv(CIKTI / "aci_duyarlilik.csv", index=False)
print(t.to_string(index=False))
