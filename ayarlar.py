"""Proje genelindeki ortak ayarlar: klasörler, kemik listesi, yer işaretleri ve ölçülen segmentler."""
from pathlib import Path

KOK = Path(__file__).resolve().parent
VERI = KOK / "veri"                      # indirilen BT maskeleri
MASKE_KLASORU = VERI / "totalsegmentator"
CIKTI = KOK / "cikti"                    # raporlar, model, grafikler
GORSELLER = KOK / "gorseller"            # öğrenci ve yapay zekâ görselleri
ISARETLER = KOK / "isaretler"            # işaretleme aracından dışa aktarılan JSON dosyaları

ZENODO_KAYIT = "10047292"                # TotalSegmentator v2.0.1, CC BY 4.0

# Omurga yayı C7–T6–T12–L5 kırık çizgisiyle ölçülür (indirilecek dosya sayısını azaltır; gerçek yayı ~%1–2 eksik ölçer)
VERTEBRALAR = ["C7", "T6", "T12", "L5"]
KEMIKLER = (
    ["skull", "sternum", "humerus_left", "humerus_right", "scapula_left", "scapula_right",
     "femur_left", "femur_right", "hip_left", "hip_right"]
    + [f"vertebrae_{v}" for v in VERTEBRALAR]
)

# 2B görsellerde işaretlenen noktalar (sıra işaretleme aracındaki sıradır)
YER_ISARETLERI = [
    ("kafatasi_1", "Kafatası en uzun çapının bir ucu (çene hariç)"),
    ("kafatasi_2", "Kafatası en uzun çapının diğer ucu"),
    ("juguler", "Sternumun üst ucu (juguler çentik)"),
    ("C7", "C7 omuru (boyun tabanı)"),
    ("L5", "L5 omuru (sakrumun hemen üstü)"),
    ("omuz_a", "Omuz eklemi (humerus başı) – figürün sağ tarafı"),
    ("dirsek_a", "Dirsek – aynı kol"),
    ("bilek_a", "El bileği – aynı kol"),
    ("omuz_b", "Omuz eklemi – figürün sol tarafı"),
    ("dirsek_b", "Dirsek – aynı kol"),
    ("bilek_b", "El bileği – aynı kol"),
    ("kalca_a", "Kalça eklemi (femur başı) – figürün sağ tarafı"),
    ("diz_a", "Diz – aynı bacak"),
    ("ayakbilegi_a", "Ayak bileği – aynı bacak"),
    ("kalca_b", "Kalça eklemi – figürün sol tarafı"),
    ("diz_b", "Diz – aynı bacak"),
    ("ayakbilegi_b", "Ayak bileği – aynı bacak"),
]

# BT modelindeki değişkenler (3B uzunluklar, mm). Sol/sağ çiftleri skorlamada değiştirilebilir sayılır.
DEGISKENLER = [
    "kafatasi_cap2",        # kafatasının ikinci büyük ana ekseni (ölçek çapası, temkinli)
    "humerus_sol", "humerus_sag",
    "femur_sol", "femur_sag",
    "juguler_omuz_sol", "juguler_omuz_sag",
    "omurga_C7_L5",         # C7–T6–T12–L5 omur merkezlerinden geçen kırık çizgi uzunluğu
    "kalca_genisligi",      # iki femur başı arası
]

# 2B segment -> (nokta1, nokta2, BT değişkeni). "a"/"b" tarafları skorlamada iki şekilde eşlenir.
SEGMENTLER = {
    "humerus_a": ("omuz_a", "dirsek_a", "humerus"),
    "humerus_b": ("omuz_b", "dirsek_b", "humerus"),
    "femur_a": ("kalca_a", "diz_a", "femur"),
    "femur_b": ("kalca_b", "diz_b", "femur"),
    "juguler_omuz_a": ("juguler", "omuz_a", "juguler_omuz"),
    "juguler_omuz_b": ("juguler", "omuz_b", "juguler_omuz"),
    "omurga_C7_L5": ("C7", "L5", "omurga_C7_L5"),
    "kalca_genisligi": ("kalca_a", "kalca_b", "kalca_genisligi"),
}
# BT etiketi olmayan, yalnızca raporlanan segmentler
EK_SEGMENTLER = {
    "onkol_a": ("dirsek_a", "bilek_a"), "onkol_b": ("dirsek_b", "bilek_b"),
    "bacak_alt_a": ("diz_a", "ayakbilegi_a"), "bacak_alt_b": ("diz_b", "ayakbilegi_b"),
}

for _k in (VERI, MASKE_KLASORU, CIKTI, GORSELLER, ISARETLER):
    _k.mkdir(parents=True, exist_ok=True)
