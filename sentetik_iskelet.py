"""BT ölçülerine göre boyutlandırılmış, rastgele pozlanmış 3B iskeletten karakalem benzeri 2B çizim üretir.

Yer işareti modelini eğitmek için kullanılır: her çizimde 17 noktanın yeri kesin olarak bilinir.
Kemik uzunlukları bt_model.json'daki BT dağılımından örneklenir; BT'de olmayan önkol ve alt bacak
uzunlukları literatürdeki oranlarla (radius/humerus ≈ 0,77; tibia/femur ≈ 0,82) türetilir.
Çizim stili (çizgi kalınlığı, gölge, kâğıt dokusu, bulanıklık) her görselde rastgele değişir
(domain randomization; Tobin vd., 2017).
"""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NOKTA_SIRASI = ["kafatasi_1", "kafatasi_2", "juguler", "C7", "L5",
                "omuz_a", "dirsek_a", "bilek_a", "omuz_b", "dirsek_b", "bilek_b",
                "kalca_a", "diz_a", "ayakbilegi_a", "kalca_b", "diz_b", "ayakbilegi_b"]
FLIP_IDX = [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]
VARSAYILAN = {"kafatasi_cap2": 150.0, "humerus": 305.0, "femur": 440.0, "juguler_omuz": 170.0,
              "omurga_C7_L5": 430.0, "kalca_genisligi": 175.0}


def rot(eksen, aci):
    eksen = np.asarray(eksen, float)
    eksen = eksen / (np.linalg.norm(eksen) + 1e-12)
    K = np.array([[0, -eksen[2], eksen[1]], [eksen[2], 0, -eksen[0]], [-eksen[1], eksen[0], 0]])
    return np.eye(3) + np.sin(aci) * K + (1 - np.cos(aci)) * K @ K


def birim(v):
    return v / (np.linalg.norm(v) + 1e-12)


def olcu_ornekle(rng, model):
    """BT modelinden bir bedenin kemik uzunluklarını örnekler (mm)."""
    if model is None:
        o = {k: v * rng.normal(1, 0.05) for k, v in VARSAYILAN.items()}
    else:
        x = rng.multivariate_normal(np.array(model["mu"]), np.array(model["sigma"]))
        L = dict(zip(model["degiskenler"], np.exp(x)))
        o = {}
        for k in VARSAYILAN:
            adaylar = [v for d, v in L.items() if d == k or d.startswith(k + "_")]
            o[k] = float(np.mean(adaylar)) if adaylar else VARSAYILAN[k] * rng.normal(1, 0.05)
    o["onkol"] = o["humerus"] * rng.normal(0.77, 0.03)
    o["alt_bacak"] = o["femur"] * rng.normal(0.82, 0.03)
    return o


class Beden:
    """Gövde çerçevesi: x = anatomik sol (+), y = yukarı, z = ön."""

    def __init__(self, rng, o, poz_cesit=1.0):
        self.rng, self.o = rng, o
        r = rng
        self.P = {}       # 3B noktalar
        self.parcalar = []  # çizim parçaları
        W, S, J = o["kalca_genisligi"], o["omurga_C7_L5"], o["juguler_omuz"]
        self.P["kalca_a"] = np.array([-W / 2, -20, 15.0])
        self.P["kalca_b"] = np.array([W / 2, -20, 15.0])

        # omurga
        n = 24
        L5 = np.array([0, 55, -45.0])
        egilme = r.choice([0, r.uniform(-0.3, 0.4), r.uniform(0.6, 1.8)], p=[0.35, 0.45, 0.2]) * poz_cesit
        yana = r.uniform(-0.4, 0.4) * poz_cesit
        burulma = r.uniform(-0.6, 0.6) * poz_cesit
        R = rot([1, 0, 0], 0.12)
        noktalar, cerceveler = [L5], [R.copy()]
        for i in range(n):
            dogal = -0.035 if i < 8 else 0.03
            R = R @ rot([1, 0, 0], dogal + egilme / n * (1.6 if i < 14 else 0.4)) @ rot([0, 0, 1], yana / n) @ rot([0, 1, 0], burulma / n)
            noktalar.append(noktalar[-1] + R @ np.array([0, S / n, 0]))
            cerceveler.append(R.copy())
        self.omurga = np.array(noktalar)
        self.omurga_R = cerceveler
        self.P["L5"], self.P["C7"] = self.omurga[0], self.omurga[-1]
        Rt = cerceveler[-1]
        self.Rt = Rt
        self.P["juguler"] = self.P["C7"] + Rt @ np.array([0, -40, 100.0])
        for taraf, sx in (("a", -1), ("b", 1)):
            yon = birim(np.array([sx * 0.93, 0.05 + r.uniform(-0.1, 0.15), -0.35]))
            self.P[f"omuz_{taraf}"] = self.P["juguler"] + Rt @ yon * J

        # baş
        Rh = Rt @ rot([1, 0, 0], r.uniform(-0.5, 0.6) * poz_cesit) @ rot([0, 0, 1], r.uniform(-0.4, 0.4) * poz_cesit) \
            @ rot([0, 1, 0], r.uniform(-0.9, 0.9) * poz_cesit)
        self.Rh = Rh
        g = o["kafatasi_cap2"]
        self.kafa_eksen = np.array([g / 2, g * 0.95 / 2, g * 1.25 / 2])   # yarı eksenler: genişlik, yükseklik, uzunluk
        self.boyun = [self.P["C7"] + Rh @ np.array([0, 18 * k, 4 * k]) for k in range(1, 7)]
        self.kafa_merkez = self.P["C7"] + Rh @ np.array([0, 130.0, 25.0])

        # kollar
        for taraf, sx in (("a", -1), ("b", 1)):
            th = np.radians(r.uniform(-5, 175)) if r.random() < 0.85 * poz_cesit + 0.05 else np.radians(r.uniform(0, 20))
            ph = np.radians(r.uniform(-40, 150))
            u = Rt @ birim(np.array([sx * np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)]))
            s = self.P[f"omuz_{taraf}"]
            e = s + u * o["humerus"]
            eksen = birim(np.cross(u, rot(u, r.uniform(0, 2 * np.pi)) @ birim(np.cross(u, [0.3, 0.2, 1]))))
            v = rot(eksen, np.radians(r.uniform(0, 150) * min(1, poz_cesit + 0.2))) @ u
            w = e + v * o["onkol"]
            self.P[f"dirsek_{taraf}"], self.P[f"bilek_{taraf}"] = e, w
            self.P[f"el_{taraf}"] = w + rot(birim(np.cross(v, [0, 0, 1]) + 1e-3), r.uniform(-0.6, 0.6)) @ v * o["humerus"] * 0.36

        # bacaklar
        destek = r.choice(["a", "b"])
        for taraf, sx in (("a", -1), ("b", 1)):
            h = self.P[f"kalca_{taraf}"]
            if taraf == destek or r.random() > 0.6 * poz_cesit:
                th, ps = np.radians(r.uniform(0, 18)), np.radians(r.uniform(-60, 200))
                diz_aci = r.uniform(0, 25)
            else:
                th, ps = np.radians(r.uniform(15, 120)), np.radians(r.uniform(-60, 200))
                diz_aci = r.uniform(0, 150)
            u = birim(np.array([sx * np.sin(th) * np.sin(ps), -np.cos(th), np.sin(th) * np.cos(ps)]))
            k = h + u * o["femur"]
            on = np.array([0, 0, 1.0])
            eksen = birim(np.cross(u, on) + 1e-6)
            v = rot(eksen, np.radians(diz_aci)) @ u
            if v @ on > u @ on + 1e-6:
                v = rot(eksen, -np.radians(diz_aci)) @ u
            a = k + v * o["alt_bacak"]
            if r.random() < 0.4:      # parmak ucu (bale)
                f = rot(eksen, -np.radians(r.uniform(5, 30))) @ v
            else:
                f = birim(on - (on @ v) * v)
            self.P[f"diz_{taraf}"], self.P[f"ayakbilegi_{taraf}"] = k, a
            self.P[f"ayak_{taraf}"] = a + f * o["alt_bacak"] * 0.62
            self.P[f"topuk_{taraf}"] = a - f * 45

    def dondur(self, Rk):
        """Kamera dönüşü: tüm 3B noktaları döndürür."""
        self.Rk = Rk


def _ciz_parcalar(b, rng):
    """Bedenin çizim parçalarını kamera koordinatlarında üretir: (derinlik, tür, veri)."""
    Rk = b.Rk
    C = lambda p: Rk @ np.asarray(p, float)
    o = b.o
    parcalar = []

    def kemik(p, q, gen, uc=1.6):
        parcalar.append(("kemik", C(p), C(q), gen, uc))

    def cizgi(pts, gen):
        parcalar.append(("cizgi", [C(p) for p in pts], gen))

    def dolgu(pts):
        parcalar.append(("dolgu", [C(p) for p in pts]))

    def yuvarlak(p, r_):
        parcalar.append(("yuvarlak", C(p), r_))

    P = b.P
    for t in "ab":
        kemik(P[f"omuz_{t}"], P[f"dirsek_{t}"], o["humerus"] * 0.075)
        dik = birim(np.cross(P[f"bilek_{t}"] - P[f"dirsek_{t}"], [0.2, 0.3, 1])) * 9
        kemik(P[f"dirsek_{t}"] + dik, P[f"bilek_{t}"] + dik, o["onkol"] * 0.05, 1.3)
        kemik(P[f"dirsek_{t}"] - dik, P[f"bilek_{t}"] - dik, o["onkol"] * 0.045, 1.3)
        # el ve parmaklar
        w, e = P[f"bilek_{t}"], P[f"el_{t}"]
        avuc = w + (e - w) * 0.45
        yan = birim(np.cross(e - w, [0.1, 1, 0.3])) * 22
        for i, f in enumerate(np.linspace(-1, 1, 5)):
            taban = avuc + yan * f
            uc = e + yan * f * 1.4 + (e - w) * rng.uniform(-0.15, 0.1) * (0.6 if i in (0, 4) else 1)
            cizgi([w + yan * f * 0.4, taban, taban + (uc - taban) * 0.5, uc], 4.5)
        kemik(P[f"kalca_{t}"], P[f"diz_{t}"], o["femur"] * 0.065, 1.7)
        dik = birim(np.cross(P[f"ayakbilegi_{t}"] - P[f"diz_{t}"], [1, 0.2, 0.1])) * 14
        kemik(P[f"diz_{t}"], P[f"ayakbilegi_{t}"], o["alt_bacak"] * 0.06, 1.5)
        kemik(P[f"diz_{t}"] + dik, P[f"ayakbilegi_{t}"] + dik, o["alt_bacak"] * 0.025, 1.3)
        a, ay, tp = P[f"ayakbilegi_{t}"], P[f"ayak_{t}"], P[f"topuk_{t}"]
        yan = birim(np.cross(ay - a, [0, 1, 0.01])) * 28
        dolgu([tp, a + yan * 0.5, ay + yan, ay - yan, a - yan * 0.5])
        for f in np.linspace(-1, 1, 5):
            cizgi([a + (ay - a) * 0.5 + yan * f * 0.8, ay + yan * f, ay + yan * f + (ay - a) * 0.2], 4)
        # klavikula, skapula
        j, s = P["juguler"], P[f"omuz_{t}"]
        orta = (j + s) / 2 + b.Rt @ np.array([0, 12, 10.0])
        cizgi([j, orta, s], 9)
        sx = -1 if t == "a" else 1
        dolgu([s + b.Rt @ np.array([-sx * 20, -15, -55.0]), s + b.Rt @ np.array([-sx * 95, -5, -70.0]),
               s + b.Rt @ np.array([-sx * 75, -150, -60.0])])
        # pelvis kanadı
        R0 = b.omurga_R[0]
        h = P[f"kalca_{t}"]
        W = o["kalca_genisligi"]
        dolgu([np.array([sx * 25, 60, -60.0]), np.array([sx * 0.62 * W, 120, -25.0]), np.array([sx * 0.66 * W, 75, 35.0]),
               h + np.array([sx * 15, 15, 0.0]), np.array([sx * 20, -45, 55.0]), np.array([sx * 30, 0, -10.0])])
        yuvarlak(h, o["femur"] * 0.06)
    dolgu([np.array([-28, 62, -62.0]), np.array([28, 62, -62.0]), np.array([0, -25, -75.0])])
    # omurga ve kaburgalar
    for i, (p, R) in enumerate(zip(b.omurga, b.omurga_R)):
        yuvarlak(p + R @ np.array([0, 0, -10.0]), 13 if i < 8 else 11)
    for p in b.boyun:
        yuvarlak(p, 9)
    toraks = list(range(9, 24))
    secilen = np.linspace(toraks[0], toraks[-1], 12).astype(int)[::-1]
    boy = o["juguler_omuz"] / 170
    for k, idx in enumerate(secilen, 1):
        p, R = b.omurga[idx], b.omurga_R[idx]
        rx = (55 + 75 * min(k, 7) / 7 - (6 * (k - 8) if k > 8 else 0)) * boy
        Dz = 62 * boy
        bitis = np.pi if k <= 7 else (0.85 * np.pi if k <= 10 else 0.55 * np.pi)
        for sx in (-1, 1):
            s = np.linspace(0, bitis, 14)
            pts = [p + R @ np.array([sx * rx * np.sin(t), -55 * t / np.pi * boy, -20 + Dz * (1 - np.cos(t))]) for t in s]
            cizgi(pts, 8)
    jug = P["juguler"]
    cizgi([jug, jug + b.Rt @ np.array([0, -90, 8.0]), jug + b.Rt @ np.array([0, -175, 12.0])], 16)
    parcalar.append(("kafa", C(b.kafa_merkez), Rk @ b.Rh @ np.diag(b.kafa_eksen), Rk @ b.Rh))
    return parcalar


def ciz(b, rng, boyut=(512, 768), ss=2):
    """Bedeni çizer; görüntü ve 17 noktanın 2B koordinatlarını döndürür."""
    parcalar = _ciz_parcalar(b, rng)
    Rk = b.Rk
    noktalar3 = {k: Rk @ b.P[k] for k in ("juguler", "C7", "L5", "omuz_a", "dirsek_a", "bilek_a", "omuz_b",
                                          "dirsek_b", "bilek_b", "kalca_a", "diz_a", "ayakbilegi_a",
                                          "kalca_b", "diz_b", "ayakbilegi_b", "ayak_a", "ayak_b", "el_a", "el_b")}
    kafa = [p for p in parcalar if p[0] == "kafa"][0]
    M = kafa[2][:2]
    U, Sv, _ = np.linalg.svd(M)
    kc = kafa[1][:2]
    k1, k2 = kc + U[:, 0] * Sv[0], kc - U[:, 0] * Sv[0]
    if k1[0] > k2[0]:
        k1, k2 = k2, k1
    tum = np.array([v[:2] for v in noktalar3.values()] + [kc + U @ (Sv * [np.cos(t), np.sin(t)]) for t in np.linspace(0, 6.3, 16)])
    tum[:, 1] *= -1
    mn, mx = tum.min(0), tum.max(0)
    W, H = boyut
    kenar = rng.uniform(0.04, 0.2)
    olcek = min(W * (1 - 2 * kenar) / (mx[0] - mn[0] + 1e-6), H * (1 - 2 * kenar) / (mx[1] - mn[1] + 1e-6))
    kaydir = np.array([W, H]) / 2 - (mn + mx) / 2 * olcek + rng.normal(0, 0.04, 2) * [W, H]

    def p2(v):
        return (np.array([v[0], -v[1]]) * olcek + kaydir) * ss

    stil = {"kalem": rng.uniform(1.0, 2.6) * ss * olcek / 0.6, "gri": int(rng.uniform(0, 70)),
            "dolgu": int(rng.uniform(205, 255)), "golge": rng.random() < 0.85, "titreme": rng.uniform(0, 1.5) * ss,
            "kalinlik": rng.uniform(1.2, 2.0)}
    img = Image.new("L", (W * ss, H * ss), 255)
    d = ImageDraw.Draw(img)
    kalem = max(1, int(stil["kalem"] / (olcek / 0.6) * rng.uniform(1.0, 2.2)))
    oz = lambda x: x * olcek * ss

    def titret(pts):
        return [tuple(np.array(p) + rng.normal(0, stil["titreme"], 2)) for p in pts]

    def derinlik(p):
        if p[0] == "kemik":
            return (p[1][2] + p[2][2]) / 2
        if p[0] in ("cizgi", "dolgu"):
            return float(np.mean([q[2] for q in p[1]]))
        return p[1][2]

    for p in sorted(parcalar, key=derinlik):
        tur = p[0]
        if tur == "kemik":
            a, c, gen, uc = p2(p[1]), p2(p[2]), oz(p[3]) * stil["kalinlik"], p[4]
            yon = birim(c - a)
            n_ = np.array([-yon[1], yon[0]])
            poli = [a + n_ * gen / 2, c + n_ * gen / 2, c - n_ * gen / 2, a - n_ * gen / 2]
            d.polygon(titret(poli), fill=stil["dolgu"], outline=stil["gri"], width=kalem)
            for q in (a, c):
                r_ = gen * uc / 2
                d.ellipse([q[0] - r_, q[1] - r_ * 0.8, q[0] + r_, q[1] + r_ * 0.8], fill=stil["dolgu"], outline=stil["gri"], width=kalem)
            d.line([tuple(a + yon * gen * 0.6), tuple(c - yon * gen * 0.6)], fill=stil["gri"] + 60, width=max(1, kalem // 2))
            if stil["golge"]:
                for t in np.linspace(0.15, 0.85, int(rng.integers(5, 14))):
                    q = a + (c - a) * t + n_ * gen * 0.3
                    d.line([tuple(q), tuple(q - n_ * gen * 0.25 + yon * gen * 0.2)], fill=stil["gri"] + 40, width=max(1, kalem // 2))
        elif tur == "cizgi":
            pts = titret([p2(q) for q in p[1]])
            gen = max(1, int(oz(p[2]) * stil["kalinlik"] * 0.6 + kalem))
            d.line(pts, fill=stil["gri"], width=gen, joint="curve")
            if gen > 4 * ss:
                d.line(pts, fill=stil["dolgu"], width=max(1, gen - 2 * kalem), joint="curve")
        elif tur == "dolgu":
            pts = titret([p2(q) for q in p[1]])
            d.polygon(pts, fill=stil["dolgu"] - 10, outline=stil["gri"], width=kalem)
        elif tur == "yuvarlak":
            q, r_ = p2(p[1]), oz(p[2]) * stil["kalinlik"]
            d.ellipse([q[0] - r_, q[1] - r_ * 0.75, q[0] + r_, q[1] + r_ * 0.75], fill=stil["dolgu"], outline=stil["gri"], width=kalem)
        elif tur == "kafa":
            c2 = p2(p[1])
            pts = [tuple(c2 + np.array([1, -1]) * (U @ (Sv * [np.cos(t), np.sin(t)])) * olcek * ss) for t in np.linspace(0, 2 * np.pi, 48)]
            # çene
            Rh = p[3]
            ax = b.kafa_eksen
            cene = [p2(p[1] + Rh @ np.array([s * ax[0] * 0.75, -ax[1] * 0.45, ax[2] * 0.25])) for s in (-1,)] + \
                   [p2(p[1] + Rh @ np.array([s * ax[0] * 0.5, -ax[1] * 1.15, ax[2] * 0.55])) for s in (-1, 1)] + \
                   [p2(p[1] + Rh @ np.array([ax[0] * 0.75, -ax[1] * 0.45, ax[2] * 0.25]))]
            d.polygon([tuple(q) for q in cene], fill=stil["dolgu"], outline=stil["gri"], width=kalem)
            d.polygon(titret(pts), fill=stil["dolgu"], outline=stil["gri"], width=kalem)
            yuz = Rh @ np.array([0, 0, 1.0])
            if yuz[2] > -0.1:
                for s in (-1, 1):
                    q = p2(p[1] + Rh @ np.array([s * ax[0] * 0.38, ax[1] * 0.05, ax[2] * 0.85]))
                    r_ = oz(ax[0] * 0.22)
                    d.ellipse([q[0] - r_, q[1] - r_ * 0.8, q[0] + r_, q[1] + r_ * 0.8], fill=stil["gri"])
                q = p2(p[1] + Rh @ np.array([0, -ax[1] * 0.3, ax[2] * 0.95]))
                r_ = oz(ax[0] * 0.1)
                d.polygon([(q[0], q[1] - r_ * 1.5), (q[0] - r_, q[1] + r_), (q[0] + r_, q[1] + r_)], fill=stil["gri"])
                dis = [p2(p[1] + Rh @ np.array([s * ax[0] * 0.4, -ax[1] * 0.62, ax[2] * 0.8])) for s in (-1, 1)]
                d.line([tuple(q) for q in dis], fill=stil["gri"], width=kalem * 2)

    img = img.resize((W, H), Image.LANCZOS)
    # kâğıt dokusu ve bozulmalar
    arr = np.asarray(img, float)
    arr += rng.normal(0, rng.uniform(0, 12), arr.shape)
    if rng.random() < 0.5:
        x = np.linspace(0, 1, W)[None, :] * rng.uniform(-30, 30) + np.linspace(0, 1, H)[:, None] * rng.uniform(-30, 30)
        arr += x - x.mean()
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    if rng.random() < 0.3:      # yüksek kontrastlı tarama görünümü
        a2 = np.asarray(img, float)
        a2 = np.clip((a2 - 128) * rng.uniform(1.5, 3.0) + 160, 0, 255)
        img = Image.fromarray(a2.astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    kp = {"kafatasi_1": k1, "kafatasi_2": k2}
    for k, v in noktalar3.items():
        kp[k] = v[:2]
    out = {}
    for k in NOKTA_SIRASI:
        v = kp[k]
        out[k] = (np.array([v[0], -v[1]]) * olcek + kaydir)
    kutu = np.array([np.array([q[0], -q[1]]) * olcek + kaydir for q in tum * [1, -1]])
    return img, out, (kutu.min(0), kutu.max(0))


def rastgele_kamera(rng):
    return rot([0, 0, 1], np.radians(rng.uniform(-12, 12))) @ rot([1, 0, 0], np.radians(rng.uniform(-20, 20))) \
        @ rot([0, 1, 0], rng.uniform(0, 2 * np.pi))


def _tek(rng, model, boyut):
    b = Beden(rng, olcu_ornekle(rng, model), poz_cesit=rng.choice([0.3, 1.0], p=[0.2, 0.8]))
    b.dondur(rastgele_kamera(rng))
    return ciz(b, rng, boyut)


def ornek(rng, model=None, boyut=(512, 768)):
    """Tek figür döndürür (geriye uyumluluk)."""
    return _tek(rng, model, boyut)


def ornek_coklu(rng, model=None, boyut=(512, 768)):
    """Ana figür ve %40 olasılıkla kenarda kısmen görünen komşu figürler (serideki kırpımlar gibi)."""
    img, kp, kutu = _tek(rng, model, boyut)
    figurler = [(kp, kutu)]
    arr = np.asarray(img, float)
    W, H = boyut
    for yon in (-1, 1):
        if rng.random() < 0.4:
            img2, kp2, kutu2 = _tek(rng, model, boyut)
            dx = int(yon * W * rng.uniform(0.55, 0.85))
            kayik = np.full_like(arr, 255.0)
            a2 = np.asarray(img2, float)
            if dx > 0:
                kayik[:, dx:] = a2[:, :W - dx]
            else:
                kayik[:, :W + dx] = a2[:, -dx:]
            arr = np.minimum(arr, kayik)
            kp2 = {k: v + np.array([dx, 0]) for k, v in kp2.items()}
            figurler.append((kp2, (kutu2[0] + [dx, 0], kutu2[1] + [dx, 0])))
    return Image.fromarray(arr.astype(np.uint8)), figurler


def model_oku(yol):
    try:
        return json.loads(open(yol, encoding="utf-8").read())
    except Exception:
        return None
