# Işıklı Masadan Üretken Yapay Zekâya: İskelet İllüstrasyonlarının Anatomik Gerçekçiliği

Bu depo, *Öğrencinin Gözünden – Lise Öğrencileri Araştırma Dergisi*'ne gönderilen çalışmanın analiz kodunu, yer işaretlerini ve sonuçlarını içerir. Atlas ve ışıklı masayla çizilmiş altı pozluk bir karakalem iskelet serisi ile aynı pozlar için ChatGPT ve Gemini ile üretilen görseller, açık kaynaklı BT verisinden (TotalSegmentator) türetilen bir gerçekçilik skoruyla karşılaştırılır.

*Analysis code, landmarks and results for a student study comparing an analog skeleton drawing series with AI-generated skeleton images, using a CT-based (TotalSegmentator) Monte Carlo realism score. Text in Turkish.*

## Depodaki dosyalar

| Klasör / dosya | İçerik |
|---|---|
| `01`–`09_*.py`, `skor_cekirdek.py`, `sentetik_iskelet.py`, `ayarlar.py` | Analiz kodu |
| `isaretleme_araci.html`, `isaretleme_rehberi.png` | Tarayıcıda çalışan yer işaretleme aracı ve 17 noktanın rehberi |
| `gorseller/ogrenci/` | Öğrenci çizimleri (P1–P6; 4× bikübik büyütülmüş). Orijinal kırpımlar: `gorseller/ogrenci_orijinal/` |
| `gorseller/yz/`, `gorseller/kayit.csv` | Yapay zekâ görselleri ve üretim kaydı (model, tarih, koşul, poz) |
| `isaretler/insan.json` | Yazarın kör modda elle koyduğu yer işaretleri (birincil veri) |
| `isaretler/otomatik_yolo_deneme.json` | Sentetik veriyle eğitilen YOLO11 modelinin tahminleri (başarısız deneme; skorlamada kullanılmaz) |
| `EKLER.md` | Makalenin ekleri: yapay zekâ istemleri (Ek A) ve teknik ayrıntılar (Ek B) |
| `cikti/` | BT referansı, model, doğrulama raporu, skorlar ve `rapor.html` |

BT maskeleri (`veri/`) depoya eklenmemiştir; `01_veri_indir.py` ile Zenodo'dan (kayıt 10047292) indirilir.

## Lisans

Kod MIT lisansıyla paylaşılır. Öğrenci çizimlerinin hakları yazara aittir. Yapay zekâ görselleri ChatGPT ve Gemini ile üretilmiştir. BT verisi: Wasserthal vd. (2023), TotalSegmentator, CC BY 4.0.

---

# İskelet Gerçekçilik Projesi

Karakalem iskelet serisi (VIVERE-MCXXXVI) ile yapay zekâ görsellerini, açık kaynak BT verisiyle eğitilmiş bir modelle karşılaştırır.

## Bu gece yapılacaklar (sırayla)

### 1. BT verisi ve model (tüm BT'lerle yeniden ölçmek için `calistir_1b_tum_bt_yeniden.bat`) — `calistir_1_bt_ve_model.bat` dosyasına çift tıkla
Python 3.10 veya üstü gerekir. Betik şunları yapar:
1. Paketleri kurar.
2. TotalSegmentator v2.0.1 veri setinden (Zenodo, CC BY 4.0) ilk 500 BT'nin yalnızca gerekli kemik maskelerini indirir (her BT için 14 küçük dosya; zip'in tamamı indirilmez).
3. Her BT'de kemik uzunluklarını ölçer (kesik kemikler dışarıda bırakılır).
4. Modeli eğitir, 5 katlı çapraz doğrulama yapar ve `cikti/bt_referans_raporu.md` dosyasını yazar.
5. Sistemi BT bedenleriyle doğrular: `cikti/dogrulama_raporu.md`.

İndirme çok yavaşsa (ekranda kalan süre 2 saati aşıyorsa): Ctrl+C ile durdur, zip'i https://zenodo.org/records/10047292 adresinden tarayıcıyla indir ve şunu çalıştır:
```
py 01_veri_indir.py --yerel-zip "C:\yol\Totalsegmentator_dataset_v201.zip"
```
Sonra 02, 03, 04 betiklerini sırayla çalıştır.

### 2. Görseller
- `gorseller/ogrenci/`: P1–P6 kırpımları hazır. Bunlar makaledeki düşük çözünürlüklü görselden kırpıldı. Mümkünse seriyi 600 dpi tarayıp aynı adlarla değiştir.
- `gorseller/yz/`: yapay zekâ görselleri. Dosya adı kuralı `MODEL_KOSUL_POZ_rN.png`, örnek `gemini_A_P3_r2.png`.
  - Web arayüzünden: İskelet Çalışması Protokolü sayfasındaki istem oluşturucuyu kullan, her görseli yeni bir sohbette üret.
  - API ile (isteğe bağlı): `06_gorsel_uret_api.py` (anahtar gerekir).

Bu gece için gerçekçi alt küme: yalnızca Koşul A, 2 model × 6 poz × 2 tekrar = 24 görsel + 6 öğrenci çizimi. Tamamı (240 görsel) sonra eklenebilir; betikler aynı kalır.

### 3. İşaretleme — `isaretleme_araci.html` dosyasını tarayıcıda aç
- **Görsel ekle** ile `gorseller/ogrenci` ve `gorseller/yz` içindeki tüm görselleri seç.
- **Kör mod**u aç (dosya adları gizlenir, sıra karışır).
- Her görselde 17 noktayı sırayla tıkla. Görünmeyen noktada **Atla** (boşluk tuşu). Geri almak için Ctrl+Z.
- Kaburga sayısı, lomber omur ve parmak sayısını gir. Tek ve tam iskelet olmayan görselde **Talimat dışı** işaretle.
- **JSON indir** ile kaydet ve dosyayı `isaretler/` klasörüne taşı.
Görsel başına yaklaşık 1–2 dakika sürer. Çalışma tarayıcıda otomatik saklanır.

### 3b. Otomatik işaretleme (insan girdisi olmadan) — `calistir_3_otomatik_isaretleme.bat`
PyTorch henüz Python 3.15'i desteklemediği için betik yapay zekâ kısmını ayrı bir Python 3.12 ortamında (`.venv_yz`) çalıştırır; Python 3.12 yoksa winget ile kurar. İlk çalıştırmada ~3 GB indirir. NVIDIA ekran kartıyla eğitim ~2 saat sürer (60 epoch, iyileşme durursa erken biter).
1. `07_sentetik_veri.py`: BT modelinden örneklenen kemik uzunluklarıyla rastgele pozlanmış 3B iskeletler kurar ve karakalem benzeri 6000 eğitim + 600 doğrulama çizimi üretir. Her çizimde 17 noktanın yeri kesin olarak bilinir.
2. `08_nokta_modeli_egit.py`: YOLO-pose modelini (COCO insan pozu ağırlıklarından) bu 17 nokta için ince ayarla eğitir. NVIDIA ekran kartı varsa onu kullanır.
3. `09_otomatik_isaretle.py`: öğrenci ve yapay zekâ görsellerini işaretler, `isaretler/otomatik.json` yazar. Elle işaretlenmiş görseller varsa model ile insan arasındaki uyumu `cikti/nokta_uyumu.md` dosyasına raporlar.
4. `05_skorla.py --isaret otomatik` raporu üretir.
Otomatik işaretleri `isaretleme_araci.html` içinde **JSON yükle** ile açıp kontrol edebilirsin. Makale için öneri: görsellerin bir kısmını elle de işaretle; model–insan uyumu yöntemin geçerlik kanıtı olur.

### 4. Rapor — `calistir_2_rapor.bat`
`cikti/rapor.html` açılır (varsayılan: `isaretler/otomatik.json` varsa otomatik işaretler, yoksa elle yapılanlar; `--isaret insan` ile değiştirilebilir): kaynak bazında skorlar, istatistik testleri, kemik bazında aşım haritası, poz grafiği.

## Yöntem (kısaca)
- **Referans:** BT maskelerinden humerus ve femur uzunluğu, juguler çentik–omuz uzaklığı, C7–L5 omurga yayı, iki femur başı arası ve kafatasının ana çapları ölçülür.
- **Model:** log uzunluklar için çok değişkenli Gauss dağılımı. BT'lerin çoğu tüm vücudu kapsamadığı için eksik veriyle EM algoritması kullanılır.
- **Skor:** Bir kemik 2B'de perspektif nedeniyle kısa görünebilir ama gerçek uzunluğundan uzun görünemez. Kafatası (yaklaşık küresel) ölçek çapasıdır; temkinli olmak için ikinci büyük ana çapı kullanılır. Model, çizimdeki uzunlukları açıklayabilecek en olası gerçek bedeni arar. Bu bedenin BT ortalamasına Mahalanobis uzaklığı **d²** skordur: 0, bu çizimle uyumlu gerçek bir insan bedeni olduğu anlamına gelir. p < 0,05 ise görsel "olanaksız" sayılır.
- **Birincil skor (iki yönlü Monte Carlo):** BT modelinden 40.000 beden örneklenir; kemikler görüntü düzlemine en fazla 45° eğik varsayılarak 2B'ye izdüşürülür, kafatası rastgele döndürülür. Yalnızca boyuna segmentler (humerus, femur, C7–L5) kullanılır; omuz ve kalça genişlikleri gövdenin dönüşüne bağlı olduğu için (yandan görünümde sıfıra iner) skora girmez, raporda ayrıca gösterilir. Görselin log(kemik / kafatası çapı) oranlarının bu dağılıma Mahalanobis uzaklığı hesaplanır; simülasyonların %5'inden daha sıra dışıysa (p < 0,05) görsel "gerçek dışı" sayılır. Fazla uzun ve fazla kısa kemikler birlikte yakalanır.
- **İkincil skor (tek yönlü üst sınır):** yukarıdaki d² testi; yalnızca fiziksel olarak olanaksız uzunlukları yakalar, gücü düşüktür.
- **Doğrulama:** Gerçek BT bedenleri rastgele açılardan 2B'ye izdüşürülür (yanlış alarm oranı). Bir kemik %15, %30 veya %50 uzatılarak yakalama gücü ölçülür.

## Sınırlılıklar (makalede yazılmalı)
- TotalSegmentator açık veri setinde radius, ulna, tibia, fibula, el ve ayak kemikleri yok. Önkol ve alt bacak yalnızca oran olarak raporlanır, skora girmez.
- Tek yönlü test yalnızca fazla uzun kemikleri yakalar; bu yüzden birincil skor iki yönlü MC skorudur. MC skoru "çizim açısı ≤ 45°" varsayımına dayanır; makalede açıkça yazılmalı.
- Doğrulama sonucu (MC, boyuna segmentler, gerçek BT n=11): hatasız bedenlerde yanlış alarm %5,5; kemik %30 kısa → %48, %30 uzun → %10, %50 uzun → %30 yakalanır. Duyarlılık analizi: kemik yönleri tamamen rastgele varsayıldığında hiçbir görsel gerçek dışı çıkmaz; 'kısa' bulgular güçlü perspektif kısalmasıyla da açıklanabilir. Yöntem kaba oran hatalarını yakalar, küçük hataları (%15) yakalayamaz.
- BT'lerin çoğu tüm vücudu kapsamadığından tam uzuv ölçümü az (n ≈ 12–18); kafatası ile femurun aynı BT'de bulunması nadirdir.
- Yer işaretleri otomatik modelle (veya elle) işaretlenir; model yalnızca sentetik çizimlerle eğitildiğinden gerçek görsellerde hata yapabilir. Kör mod kullan; mümkünse ikinci bir kişi görsellerin bir kısmını bağımsız işaretlesin (değerlendiriciler arası uyum).
- Kaburga ve omur sayıları anatomik norma (12 çift kaburga, 5 lomber omur) göre değerlendirilir.

## Klasörler
| Klasör | İçerik |
|---|---|
| `veri/totalsegmentator/` | indirilen BT kemik maskeleri |
| `gorseller/` | öğrenci ve yapay zekâ görselleri, `kayit.csv` |
| `isaretler/` | işaretleme aracından JSON |
| `cikti/` | model, raporlar, şekiller |

Kaynak: Wasserthal, J. vd. (2023). TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. *Radiology: Artificial Intelligence, 5*(5), e230024. Veri seti CC BY 4.0.
