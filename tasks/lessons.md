# Dersler

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
