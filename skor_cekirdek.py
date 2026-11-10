"""BT verisiyle eğitilmiş modelden bir görselin anatomik olabilirlik skorunu hesaplayan çekirdek.

Fikir: Bir kemiğin 2B görüntüdeki uzunluğu, perspektif kısalması nedeniyle 3B uzunluğundan kısa olabilir,
ama uzun olamaz. Kafatası yaklaşık küresel olduğundan ölçek çapası olarak kullanılır. Model, çizilen
uzunlukları açıklayabilecek en olası gerçek bedeni (BT dağılımı içinde) arar. Bulunan bedenin
ortalamadan Mahalanobis uzaklığı (d²) skordur: 0 = tamamen olası, büyüdükçe anatomik olarak olanaksız.
"""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from ayarlar import CIKTI, SEGMENTLER

CAPA = "kafatasi_cap2"


def model_yukle(yol=CIKTI / "bt_model.json"):
    m = json.loads(yol.read_text())
    m["mu"], m["sigma"] = np.array(m["mu"]), np.array(m["sigma"])
    return m


def _degisken(seg_tipi, taraf):
    if seg_tipi in ("omurga_C7_L5", "kalca_genisligi"):
        return seg_tipi
    return f"{seg_tipi}_{taraf}"


def _coz(model, kisitlar):
    """kisitlar: {degisken: c} ; x_degisken - x_capa >= c (log uzayında)."""
    adlar = [CAPA] + list(kisitlar)
    idx = [model["degiskenler"].index(a) for a in adlar]
    mu, S = model["mu"][idx], model["sigma"][np.ix_(idx, idx)]
    P = np.linalg.inv(S)
    A = np.zeros((len(kisitlar), len(adlar)))
    for i in range(len(kisitlar)):
        A[i, 0], A[i, i + 1] = -1, 1
    c = np.array(list(kisitlar.values()))
    if np.all(A @ mu - c >= 0):
        return 0.0, mu
    f = lambda x: (x - mu) @ P @ (x - mu)
    g = lambda x: 2 * P @ (x - mu)
    r = minimize(f, mu, jac=g, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda x: A @ x - c, "jac": lambda x: A}],
                 options={"maxiter": 200, "ftol": 1e-10})
    return float(max(r.fun, 0.0)), r.x


def oran_siniri(model, degisken, yuzdelik=0.975):
    """BT'de (degisken / kafatasi_cap2) oranının üst yüzdeliği."""
    i, j = model["degiskenler"].index(degisken), model["degiskenler"].index(CAPA)
    mu = model["mu"][i] - model["mu"][j]
    var = model["sigma"][i, i] + model["sigma"][j, j] - 2 * model["sigma"][i, j]
    return float(np.exp(mu + norm.ppf(yuzdelik) * np.sqrt(var))), float(np.exp(mu))


def skorla(model, uzunluklar_2b, kafatasi_2b):
    """uzunluklar_2b: {segment_adi: piksel}, kafatasi_2b: kafatasının en uzun çizilmiş çapı (piksel)."""
    if not kafatasi_2b or kafatasi_2b <= 0:
        return None
    en_iyi = None
    for ters in (False, True):                 # a/b tarafını sol/sağ ile iki şekilde eşle
        kisitlar, segler = {}, {}
        for seg, l in uzunluklar_2b.items():
            if seg not in SEGMENTLER or not l or l <= 0:
                continue
            tip = SEGMENTLER[seg][2]
            taraf = None
            if seg.endswith(("_a", "_b")):
                harita = {"a": "sol", "b": "sag"} if ters else {"a": "sag", "b": "sol"}
                taraf = harita[seg[-1]]
            d = _degisken(tip, taraf)
            if d not in model["degiskenler"]:
                continue
            kisitlar[d] = np.log(l / kafatasi_2b)
            segler[d] = seg
        if not kisitlar:
            return None
        d2, _ = _coz(model, kisitlar)
        if en_iyi is None or d2 < en_iyi[0]:
            en_iyi = (d2, kisitlar, segler)
    d2, kisitlar, segler = en_iyi
    k = len(kisitlar)
    ayrinti = {}
    for d, c in kisitlar.items():
        ust, medyan = oran_siniri(model, d)
        oran = float(np.exp(c))
        ayrinti[segler[d]] = {"oran": oran, "bt_medyan": medyan, "bt_ust_975": ust,
                              "asim": oran > ust, "medyana_gore": oran / medyan}
    return {"d2": d2, "k": k, "olasilik": float(chi2.sf(d2, k)) if d2 > 0 else 1.0,
            "olanaksiz": bool(chi2.sf(d2, k) < 0.05) if d2 > 0 else False,
            "asim_sayisi": int(sum(v["asim"] for v in ayrinti.values())), "segmentler": ayrinti}


# ---------------------------------------------------------------------------
# İki yönlü Monte Carlo gerçekçilik skoru
# BT modelinden çok sayıda beden örneklenir ve çizim benzeri açılardan 2B'ye izdüşürülür. Bir görselin
# kemik/kafatası oranları bu "gerçek bedenlerin çizimleri" dağılımıyla karşılaştırılır. Hem fazla uzun hem
# fazla kısa kemikler yakalanır. Kafatası geometrisi sentetik_iskelet.py'deki çizimle aynıdır
# (yarı eksenler: g/2, 0,95g/2, 1,25g/2; g = BT kafatası ikinci ana çapı), böylece otomatik işaretlerle tutarlıdır.
# ---------------------------------------------------------------------------
_SIM_ONBELLEK = {}
TIP_DEG = {"humerus": ("humerus_sol", "humerus_sag"), "femur": ("femur_sol", "femur_sag"),
           "juguler_omuz": ("juguler_omuz_sol", "juguler_omuz_sag"),
           "omurga_C7_L5": ("omurga_C7_L5",), "kalca_genisligi": ("kalca_genisligi",)}
# Boyuna segmentler (uzun kemikler ve omurga) çizimlerde görüntü düzlemine yakın durur; ≤45° varsayımı onlar için
# makuldür. Enine genişlikler (omuz, kalça) ise gövdenin dönüşüne bağlıdır: yandan görünümde sıfıra yaklaşır. Bu
# yüzden birincil skor yalnızca boyuna segmentleri kullanır; genişlikler raporda ayrıca gösterilir.
BOYUNA = ("humerus", "femur", "omurga_C7_L5")


def _rastgele_donus(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def simulasyon(model, gorus="cizim", n=40000, tohum=0):
    """gorus='cizim': kemikler görüntü düzlemine en fazla 45° eğik (çizimlerde tipik);
    gorus='rastgele': her yön eşit olasılıklı (en temkinli)."""
    anahtar = (id(model), gorus, n)
    if anahtar in _SIM_ONBELLEK:
        return _SIM_ONBELLEK[anahtar]
    rng = np.random.default_rng(tohum)
    X = rng.multivariate_normal(model["mu"], model["sigma"], size=n)
    L = {d: np.exp(X[:, i]) for i, d in enumerate(model["degiskenler"])}
    g = L[CAPA]
    R = _rastgele_donus(rng, n)
    M = R[:, :2, :] * (np.stack([g / 2, 0.95 * g / 2, 1.25 * g / 2], 1)[:, None, :])
    D = 2 * np.linalg.svd(M, compute_uv=False)[:, 0]
    oranlar = {}
    for d, v in L.items():
        if d == CAPA:
            continue
        if gorus == "cizim":
            kis = np.cos(np.radians(rng.uniform(-45, 45, n)))
        elif isinstance(gorus, (int, float)):          # duyarlılık analizi: en büyük eğim açısı (derece)
            kis = np.cos(np.radians(rng.uniform(-gorus, gorus, n)))
        else:
            kis = np.sqrt(1 - rng.uniform(-1, 1, n) ** 2)
        oranlar[d] = np.log(v * kis / D)
    _SIM_ONBELLEK[anahtar] = oranlar
    return oranlar


def skorla_mc(model, uzunluklar_2b, kafatasi_2b, gorus="cizim", tipler=BOYUNA):
    """Görselin log(kemik/kafatası) oranlarının, BT bedenlerinin simüle çizimlerine göre ne kadar sıra dışı
    olduğunu verir. p küçükse (ör. < 0,05) görselin oranları gerçek insanların çizimlerinde nadir görülür."""
    if not kafatasi_2b or kafatasi_2b <= 0:
        return None
    sim = simulasyon(model, gorus)
    en_iyi = None
    for ters in (False, True):
        z, degs, segs = [], [], []
        for seg, l in uzunluklar_2b.items():
            if seg not in SEGMENTLER or not l or l <= 0:
                continue
            tip = SEGMENTLER[seg][2]
            if tipler and tip not in tipler:
                continue
            secenek = TIP_DEG[tip]
            if len(secenek) == 2:
                d = secenek[(seg.endswith("_a")) ^ ters]   # a -> sağ (ters değilse)
            else:
                d = secenek[0]
            if d not in sim or d in degs:
                continue
            z.append(np.log(l / kafatasi_2b))
            degs.append(d)
            segs.append(seg)
        if not z:
            return None
        Z = np.stack([sim[d] for d in degs], 1)
        m, C = Z.mean(0), np.cov(Z, rowvar=False).reshape(len(degs), len(degs))
        P = np.linalg.inv(C)
        z = np.array(z)
        d2 = float((z - m) @ P @ (z - m))
        d2_sim = np.einsum("ij,jk,ik->i", Z - m, P, Z - m)
        p = float((d2_sim >= d2).mean())
        if en_iyi is None or p > en_iyi[1]:
            sd = np.sqrt(np.diag(C))
            ayrinti = {s: {"z": float((zi - mi) / si), "medyana_gore": float(np.exp(zi - np.median(Z[:, j])))}
                       for j, (s, zi, mi, si) in enumerate(zip(segs, z, m, sd))}
            en_iyi = (d2, p, ayrinti)
    d2, p, ayrinti = en_iyi
    return {"mc_d2": d2, "mc_p": p, "mc_gercek_disi": p < 0.05, "mc_k": len(ayrinti), "mc_segmentler": ayrinti}
