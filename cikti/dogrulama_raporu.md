# Skorlama Sisteminin Doğrulanması

Çarpan 1,0 satırları gerçek bedenlerdir: buradaki 'olanaksız' oranı yanlış alarm oranıdır (düşük olmalı).
Çarpan > 1 satırlarında rastgele bir kemik uzatılmıştır: buradaki oran yakalama gücüdür (yüksek olmalı).
Perspektif kısalması nedeniyle uzatılan kemik görüntüde zaten kısa görünüyorsa hata yakalanamaz; bu yöntemin doğal sınırıdır.

| Küme | Kemik çarpanı | n | 'Olanaksız' oranı | En az bir oran aşımı |
|---|---|---|---|---|
| gerçek BT (n=11) | 1.00 | 220 | %0.0 | %0.0 |
| gerçek BT (n=11) | 1.15 | 220 | %0.0 | %0.0 |
| gerçek BT (n=11) | 1.30 | 220 | %0.0 | %0.0 |
| gerçek BT (n=11) | 1.50 | 220 | %0.0 | %1.4 |
| modelden örneklenen (n=200) | 1.00 | 4000 | %0.0 | %0.1 |
| modelden örneklenen (n=200) | 1.15 | 4000 | %0.0 | %0.5 |
| modelden örneklenen (n=200) | 1.30 | 4000 | %0.0 | %2.5 |
| modelden örneklenen (n=200) | 1.50 | 4000 | %0.4 | %10.8 |

## İki yönlü Monte Carlo gerçekçilik skoru (çizim benzeri açılar)

Kemikler görüntü düzlemine en fazla 45° eğik varsayılır. Çarpan 1,0 = hatasız (yanlış alarm, ≈%5 beklenir); 0,7 = kemik %30 kısa; 1,15–1,5 = kemik uzun.

| Küme | Kemik çarpanı | n | 'Gerçek dışı' (p<0,05) oranı |
|---|---|---|---|
| gerçek BT (n=11) | 0.70 | 220 | %48.2 |
| gerçek BT (n=11) | 1.00 | 220 | %5.5 |
| gerçek BT (n=11) | 1.15 | 220 | %5.5 |
| gerçek BT (n=11) | 1.30 | 220 | %10.0 |
| gerçek BT (n=11) | 1.50 | 220 | %30.0 |
| modelden örneklenen (n=200) | 0.70 | 4000 | %30.4 |
| modelden örneklenen (n=200) | 1.00 | 4000 | %4.1 |
| modelden örneklenen (n=200) | 1.15 | 4000 | %7.8 |
| modelden örneklenen (n=200) | 1.30 | 4000 | %19.1 |
| modelden örneklenen (n=200) | 1.50 | 4000 | %40.7 |

![Doğrulama](sekil_dogrulama.png)