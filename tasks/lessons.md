# Dersler

## 2026-08-07 — Tek makinede ölçülen sayı, tolerans değildir

Yeni transient testine `1e-10` eşiği koydum; kendi makinemde ölçüm `8.62e-12`
idi, üç basamak pay bıraktığımı sanıyordum. CI kırmızı döndü: **ubuntu 1.624e-10,
windows 1.570e-10** — aynı hesap, farklı LAPACK, tridiagonal çözümde farklı
yuvarlama yolu. Fizikte hiçbir sorun yoktu; kusur toleranstaydı.

**Kural.** Toleransı ölçtüğün sayıya göre değil, **gerçek bir kusurun neye
benzeyeceğine** göre kur. İşaret hatası, eksik bir `1/h̄` ya da yanlış Bernoulli
argümanı O(1) verir; o hâlde eşik 1e-9 olabilir ve hem platform yuvarlamasının
üç basamak üstünde durur hem de gerçek kusurun dokuz basamak altında kalır.
Mutlak yerine **büyüklüğün ölçeğine göre bağıl** yaz, ki iddia her platformda
aynı şeyi söylesin.

Bir de: ölçümü tek platformdan raporlama. Kanıt dizesine "worst case across
macOS, ubuntu ve windows" yazmak, sayının nereden geldiğini görünür kılar.

## 2026-08-07 — `cp` ile geri alma, bayat `.pyc`'yi taze gösterebilir

Bir mutasyon testinde dosyayı `cp` ile geri aldım; kaynakta doğru değer
görünüyordu ama çalışma zamanı hâlâ mutasyonlu değeri okuyordu. Sebep:
`.pyc` geçerlilik kontrolü kaynağın **mtime + boyutuna** bakar. Mutasyon aynı
uzunlukta bir sayıydı (boyut değişmedi) ve `cp` mtime'ı `.pyc`'nin kaydettiği
değere eşitledi — Python da bytecode'u taze saydı.

**Kural.** Bir dosyayı yerinde geri alıp davranışını sınayacaksan `touch` ile
mtime'ı ilerlet, ya da import'u yeni bir süreçte doğrula. Ve "test geçti"
demeden önce, testin **hangi** koda baktığını en az bir kez `module.__file__` ve
gerçek bir değerle teyit et.

Daha genel ders: mutasyon testi yalnızca mutasyonu doğrulamaz, geri almayı da
doğrulaman gerekir. Yarısı yapılmış bir mutasyon testi, yanlış bir güven verir.

## 2026-08-07 — Konuşulmuş ama kod üretmemiş karar, sıkıştırmada kaybolur

**Ne oldu.** "4D" tanımı kullanıcıyla konuşulup karara bağlandı, ama karar
yalnızca sohbette kaldı. Context sıkıştırması onu düşürdü; kullanıcı konuyu
yeniden açmak zorunda kaldı ve düzeltti: *"4D'yi konuşmuştuk context window
sıkıştırırken unutmuşsun"*.

**Örüntü.** Sıkıştırma özeti kod değişikliklerini, dosya yollarını ve hata
mesajlarını iyi taşıyor — hepsinin diskte bir izi var. **Henüz kod üretmemiş bir
kararın izi yoktur.** Dolayısıyla en kırılgan bilgi, en az korunanıdır:
uygulanmamış karar.

**Kural.** Bir tasarım kararı alındığı TURDA dosyaya yazılır. `tasks/todo.md`
içinde "Kararlar" başlığı; karar + tarih + **gerekçe**. Gerekçesiz karar sonraki
oturumda yeniden tartışmaya açılır, çünkü kimse neden öyle olduğunu bilmez.
Uygulamaya bir sonraki turda başlayacak olsan bile yaz — iki tur arasında
sıkıştırma olabilir.

**Özellikle:** `AskUserQuestion` ile alınan her cevap. Cevabı sohbette
özetlemekle yetinmek, onu kaybetmenin en kolay yoludur.

## 2026-08-07 — Bir oracle sayıyı reddettiğinde önce sayıdan şüphelen

`physics_verify` bir sayıyı reddettiğinde ilk hipotez "harness tuhaflığı" değil
"sayı yanlış" olmalı. Bu oturumda altı kez oracle haklı çıktı, sıfır kez harness
hatalıydı. Bir kez tersini varsayıp kontak potansiyelinin 7. hanesini yanlış
raporladım.

## 2026-08-07 — Tek bir olgunun iki kaynağı varsa, ikisini teste bağla

Sürüm numarası `pyproject.toml` ve `src/tarhan/__init__.py` içinde ayrı ayrı
duruyordu; kontrol listesi yalnızca birincisini işaret ediyordu. Oysa paketin
kendini tanıttığı yer ikincisidir — `cli.py` ve `mcp_server.py` onu okur.
Tesadüfen grep'lerken görmesem, sürümü kendisiyle çelişen bir paket
yayımlanacaktı.

**Kural.** Aynı olguyu iki yerde tutan her şey ya tek kaynağa indirilir ya da
aralarındaki tutarlılık bir testle çivilenir. Kontrol listesini satır numarasına
değil olguya göre yaz: "pyproject satır 10" değil, "paketin bildirdiği sürüm".

## Bir tolerans, tek makinede geçtiği için doğru değildir (2026-08-10)

**Ne oldu.** `oracle.yml` ilk kez gerçekten çalıştı ve 3 test düştü. 2D I-V
karşılaştırmasında TARHAN'ın 0.2 V'taki deşik akımı DEVSIM'e karşı CI'da 1.063,
bu dizüstünde 1.011 çıktı. DEVSIM'in kendi sayısı iki platformda BİREBİR aynıydı
(1.011424134349962) — oynayan taraf bizdik.

**Kritik adım: sebebi aramak, toleransı gevşetmek değil.** `gummel_tol`
sıkılaştırıldığında oran 1.011 → 1.011 → 1.0347 → 1.0273 (1e-9 → 1e-14) gitti,
iterasyon 3 → 27. **Monoton olmayan bir sapma, yakınsamamış demektir.** Yani
1e-9'da da yakınsamamıştı; bu makinedeki uyum şanstı, CI aynı desteden başka bir
kart çekti.

**Kural.** Bir eşik "burada geçiyor" diye doğrulanmış sayılmaz. Bir büyüklüğün
gerçekten belirlenmiş olup olmadığını sınamanın en ucuz yolu, yakınsama
ölçütünü sıkıp cevabın SABİT kalıp kalmadığına bakmaktır. Sabit kalmıyorsa
yayımlanan iddia daraltılmalı — sayı düzeltilmeli, test silinmemeli.

**Ne yapıldı.** Doğrulanmış aralık 0.3-0.5 V'a çekildi; 0.2 V hâlâ çözülüyor ve
karşılaştırılıyor ama ölçümün destekleyebildiği bir bantla, ve neden öyle olduğu
testin içinde yazıyor. README, capability registry ve DESIGN-2D §5 birlikte
düzeltildi. Elektron akımı 0.2 V'ta kararlı ve hâlâ 1e-4'e çiviliyor.

**İkinci ders: yayımlanmış bir CI dosyası çalıştırılmamış koddur.** oracle.yml
üç birim testiyle "çürümeye karşı" korunuyordu ama hiç koşmamıştı; ilk koşuda
DEVSIM'in BLAS bağımlılığı yüzünden daha ilk adımda patladı. Bir workflow'un
yazılmış olması onun çalıştığı anlamına gelmez.

## Kanıt alanını kanıta bakmadan doldurmak (2026-08-10)

**Ne oldu.** Capability kayıtlarına metrik başına "ölçülen bias noktaları"
eklerken 2D listesini testin `@parametrize` satırlarından OKUDUM, 1D listesini
ise registry'deki düzyazı kanıttan ("over 0.15-0.40 V") **türettim**:
`(0.15, 0.20, 0.25, 0.30, 0.35, 0.40)`. Oracle'ın gerçek taraması
`(0.1, 0.2, 0.3, 0.4)`. Yani hiç ölçülmemiş üç bias uydurdum ve ölçülen
0.1 V'u dışarıda bıraktım — üstelik bunu, işi "ne ölçüldü" demek olan alanın
içinde yaptım.

**Neden kaçtı.** Yazdığım test yalnızca `evidence` dosyasının VAR olduğunu
kontrol ediyordu. Dosyanın varlığı, içindeki sayıların iddia edilenler olduğunu
söylemez. Kendi kontrolüm iddianın yanından geçiyordu.

**Kural.** Bir alan "şu ölçüldü" diyorsa, değeri ölçümü üreten KODDAN
gelmelidir; ölçümü anlatan cümleden değil. Kanıt sayıları elle kopyalanacaksa,
onları üreten kaynağa karşı çivileyen bir test şart.

**Ne yapıldı.** `devsim_pn1d_compare.VOLTS` tek doğruluk kaynağı olarak dışa
açıldı ve registry ona karşı çivilendi; envelope da `[0.10, 0.40]` olarak
düzeltildi. Bunu denetleyen bulmadı — fizik dürüstlük kapısı "bu sayıyı
doğruladın mı?" diye sorunca ben kontrol ettim ve yanlış çıktı.

## Aynı hatayı, o hatayı çivileyen testin İÇİNDE yaptım (2026-08-10)

**Ne oldu.** Yakınsama bulgusunu kalıcı kılmak için yazdığım test
`marginal["current_rel_change"] > 1e-5` diyordu. Bu makinede 3.49e-04, ubuntu
CI'da 4.29e-06 çıktı ve CI kırmızı döndü. Yani eşiği yine tek makinenin ölçtüğü
sayıya göre kurmuşum — `lessons.md`'de zaten yazılı olan ders, ve o testi
tanıtan commit mesajında **kendi elimle alıntıladığım** ders.

**Neden tekrarladı.** Bulguyu "0.1 V'ta akım 6e-5 mertebesinde geziniyor" diye
hatırladım ve bu SAYIYI çiviledim. Oysa bulgu sayı değil, **fark**: bir bias
yerleşiyor, diğeri yerleşmiyor. Mutlak büyüklük LAPACK'e göre değişir; oran
değişmez (bu makinede 124667x, ubuntu'da ~4e4).

**Kural.** Bir ölçümü teste çevirirken önce şunu sor: *bulgunun kendisi hangi
niceliktir?* Mutlak bir değeri ancak bir DEFECT'in üreteceği büyüklükten
türetebiliyorsan çivile. İki durumu ayırt eden bir bulguysa, aralarındaki ORANI
çivile — oran, uygulama farklarına karşı dayanıklıdır.

**Uyarı işareti.** Bir testin içine yazdığın sayı, o testin docstring'inde
"bu makinede ölçüldü" diye geçiyorsa, o sayı muhtemelen assertion'da olmamalı.
