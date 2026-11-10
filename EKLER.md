# Makale Ekleri

Makalenin kelime sınırı nedeniyle istem metinleri ve teknik ayrıntılar bu dosyaya taşınmıştır.

## Ek A. Yapay Zekâ Görselleri İçin Kullanılan İstemler

**Koşul A (temel):** A graphite pencil drawing on plain white paper of one complete adult human skeleton {POZ}. The skeleton should be anatomically accurate, as in a medical anatomy atlas. Show the entire skeleton from the top of the skull to the tips of the toes, with nothing cropped. Bones only: no skin, no muscles, no clothing. Black and grey pencil lines with light shading, in the style of an art-school anatomy study. Plain white background with no other objects and no text. One figure only, centered, portrait orientation.

**Koşul B (ayrıntılı anatomi), Koşul A’ya eklenen metin:** Anatomical requirements: 12 pairs of ribs; 7 cervical, 12 thoracic and 5 lumbar vertebrae; a sacrum and coccyx; a separate radius and ulna in each forearm and a separate tibia and fibula in each lower leg; a patella on each knee; complete hands, each with 8 carpal bones, 5 metacarpals and 14 phalanges; complete feet, each with 7 tarsal bones, 5 metatarsals and 14 phalanges; realistic adult proportions.

**{POZ} yerine kullanılan tanımlar:**

- **P1:** standing on the toes of one foot, with the other knee raised forward and that foot drawn up beside the standing knee; one arm reaches straight up above the head and the other arm extends forward at shoulder height; seen from the side, with the spine slightly arched back

- **P2:** standing on one straight leg with the torso bent far forward and sideways so that the skull hangs low at hip height, facing the viewer; one arm reaches diagonally up and out; the other leg is bent at the knee with the foot raised behind the body and held by the other hand

- **P3:** seen from the front, standing on the toes of one foot; the other knee is raised high and turned outward with the foot resting against the standing knee; one hand rests on the hip with the elbow bent outward and the other arm is raised straight above the head; the skull is turned in three-quarter view

- **P4:** seen from the side, standing on the toes of one straight leg; the other leg is kicked forward and up to hip height with the knee straight; one arm is raised straight up above the head and the other arm reaches forward along the raised leg toward the foot

- **P5:** in a deep forward bend with both feet on the ground and the legs straight and apart; the torso hangs down so that the skull is at knee height, facing the viewer; both arms are raised up and back above the hips, one straight and one bent at the elbow

- **P6:** seen in front three-quarter view, standing with the legs crossed, one foot in front of the other and both heels raised; both hands are placed on top of the head with the elbows pointing outward; the torso is slightly twisted and the head is tilted to one side

ChatGPT arayüzünde istemin başına “Generate an image:” ifadesi eklenmiştir.

## Ek B. Yer İşaretleri ve Teknik Ayrıntılar

**Kod ve veri deposu:** https://github.com/egemenonur04/iskelet-gercekcilik. Depoda analiz kodu, tarayıcı tabanlı işaretleme aracı, elle konan yer işaretleri, tüm görseller ve üretim kaydı, BT referans ölçüleri, doğrulama raporu ve skorlar bulunmaktadır. BT maskeleri lisans ve boyut nedeniyle depoya eklenmemiş; 01_veri_indir.py ile Zenodo’dan indirilebilmektedir.

**Yer işaretleri (sırasıyla):** (1-2) kafatasının en uzun çapının iki ucu, çene hariç; (3) juguler çentik; (4) C7 omuru; (5) L5 omuru; (6-8) figürün sağ omuz, dirsek ve el bileği; (9-11) sol omuz, dirsek ve el bileği; (12-14) sağ kalça (femur başı), diz ve ayak bileği; (15-17) sol kalça, diz ve ayak bileği. Taraf belirsiz olduğunda skor iki eşleştirmeyi de dener ve daha olası olanı seçer.

**BT ölçümü:** Maskeler nibabel ile okunmuş; uzun kemiklerin uçları ana eksen boyunca %8’lik uç dilimlerin merkezleri olarak alınmıştır. Erişkin olabilirlik aralıkları (mm): kafatası ikinci ana çapı 115-180, humerus 230-400, femur 330-560, juguler–omuz 110-240, omurga 330-560, femur başları arası 130-230.

**Model ve skor:** Log uzunluklar için çok değişkenli normal dağılım; EM algoritması; kovaryansa 0,15 küçültme. Monte Carlo benzetiminde 40.000 beden; kemik eğimi -45° ile +45° arasında düzgün dağılımlı; kafatası yarı eksenleri g/2, 0,95g/2 ve 1,25g/2 (g: BT kafatası ikinci ana çapı) olan bir elipsoit olarak rastgele döndürülmüştür. p değeri, benzetimdeki d² değerlerinin görselinkine eşit ya da büyük olanlarının oranıdır.

**Otomatik yer işareti denemesi:** 6000 eğitim ve 600 doğrulama görüntüsü; YOLO11n-pose ağırlıklarından başlanarak 640 piksel çözünürlükte, 60 tur, NVIDIA ekran kartında eğitim; yatay çevirme veri artırımı.
