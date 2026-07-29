# Otobüs Yolcu Talebi Tahmini

[English documentation](README.md)

Bu depo, planlanan bir şehirlerarası otobüs seferi için beklenen yolcu talebini tahmin eder. Bu çalışma bir **talep tahmin sistemidir**; bilet fiyatı, kapasite, işletme maliyeti, komisyon ve vergiler kapsam dışındadır. Bu nedenle çıktı doğrudan kârlılık kararı olarak kullanılmamalıdır.

Model; firma, firmaya ait güzergâh kodu, sefer tarihi ve kalkış saatini alır. Sayısal talep tahmininin yanında eşik olasılıkları, talep etiketi ve geçmiş verinin gücüne dayanan güvenilirlik değerlendirmesi üretir.

> **Hedef değişken:** `SEFER_SAYISI` (bu projede yolcu talebi olarak kullanılır)<br>
> **Veri:** 2023-01-01 ile 2026-04-14 arasında 9.103.971 temizlenmiş sefer kaydı<br>
> **Mevcut servis adayı:** v4.2 hibrit sayısal tahmin + v4.4 eşik sınıflandırıcıları

## Depo yapısı

```text
training/
├── analysis.duckdb              # Eğitim ve çıkarımda kullanılan DuckDB veritabanı
├── models/                      # Kullanıma alınmış CatBoost modelleri ve eşik bilgileri
├── results/                     # v4.2 birleştirme kuralı ve deney sonuçları
├── inference/                   # Kendi başına çalışan, salt okunur tahmin sistemi
│   ├── engine.py                # Modelleri yükleme, tahmin akışı ve güvenilirlik
│   ├── features.py              # Özellik sözleşmesi ve DuckDB ayarları
│   ├── batch_features.py        # Teklifler için vektörleştirilmiş özellik üretimi
│   ├── hybrid.py                # Dondurulmuş v4.2 birleştirme kuralı
│   ├── classifiers.py           # v4.4 model ve eşik dosyalarının doğrulanması
│   ├── predict_trip.py          # Tek sefer komut satırı aracı
│   ├── predict_trips_batch.py   # CSV toplu tahmin aracı
│   ├── check_trips.py           # İnceleme bayrakları üreten denetim aracı
│   ├── stop_addition/           # Durak ekleme tahmini, artış ve iş kuralları
│   ├── artifacts/stop_addition/ # Seçili durak ekleme modeli ve sözleşmesi
│   ├── tests/stop_addition/     # Dondurulmuş 65 satırlık girdi ve referans
│   ├── api/                     # İki proje için ortak FastAPI servisi
│   └── frontend/                # Preact/Vite arayüzü
├── scripts/                     # Sürümlenmiş eğitim ve değerlendirme akışları
│   ├── shared/                  # Ortak sabitler, DuckDB ile özellik üretimi, metrikler ve model yükleme
│   ├── baseline/                # Ortak çalıştırıcıyı kullanan tarihsel ortalama taban modelleri
│   └── catboost/v1 … v4_5/      # Ortak modülleri kullanan, yeniden çalıştırılabilir model deneyleri
├── pyproject.toml               # Python bağımlılıkları ve sürüm gereksinimi
├── package.json                 # Ön yüz ve arka yüz geliştirme komutları
└── README.md                    # İngilizce dokümantasyon
```

`scripts/` dizini eğitim ve değerlendirme akışlarını içerir. Her sürüm deneyi (`v4_1` ile `v4_5` arasındakiler) ortak kodu yalnızca `scripts/shared/` dizininden alır; başka bir sürümün kodunu içe aktarması gerekmez. Ortak modüller şunlardır:

| Modül | Görevi |
|---|---|
| `scripts/shared/paths.py` | Proje dizini, veritabanı ve model yolları |
| `scripts/shared/constants.py` | Özellik sütunları, gruplama anahtarları ve kaynak sütunlar |
| `scripts/shared/period_config.py` | `PeriodConfig` veri sınıfı ile standart eğitim, doğrulama, test ve nihai dönemler |
| `scripts/shared/feature_pipeline.py` | DuckDB ile özellik üretimi: kaynak tablolar, uzun dönem istatistikleri, yakın geçmiş pencereleri, özelliklerin birleştirilmesi ve sızıntı denetimi |
| `scripts/shared/metrics.py` | Regresyon, sınıflandırma ve olasılık metrikleri |
| `scripts/shared/model_utils.py` | CatBoost modelinin yüklenmesi ve özellik sözleşmesinin doğrulanması |
| `scripts/shared/baseline_runner.py` | Parametrelerle çalışan hiyerarşik taban model çalıştırıcısı |

`GUZERGAH_KODU` firma bazlıdır; güvenli güzergâh kimliği bu yüzden `FIRMA_ID + GUZERGAH_KODU` bileşimidir. `guzergah_canonical` tablosu ayrıca firma güzergâhlarını fiziksel güzergâhlara eşler. Böylece sistem hem firmaya özgü geçmişten hem de aynı fiziksel hattı işleten diğer firmaların geçmişinden yararlanabilir.

## Kullanılan teknolojiler

Python 3.14+ ortamı [uv](https://docs.astral.sh/uv/) ile yönetilir. Python tarafında CatBoost, DuckDB, pandas, scikit-learn, FastAPI, Uvicorn ve Matplotlib kullanılır. Eğitimde CatBoost'un GPU modu (`task_type="GPU"`, cihaz `0`) kullanılır. Tahmin çalıştırmak için uyumlu bir CatBoost kurulumu ile kayıtlı model dosyalarının bulunması yeterlidir.

Web arayüzü Preact, Vite, TypeScript, Tailwind CSS ve Leaflet ile hazırlanmıştır. Harita katmanı OpenStreetMap üzerinden alınır. Bağımlılıklar `package.json` ve `bun.lock` üzerinden npm veya Bun ile yönetilir.

Milyonlarca geçmiş kayıt üzerinde filtre, birleştirme, grup istatistiği, quantile ve özellik üretimi DuckDB içinde yapılır. Pandas yalnızca tamamlanmış model matrisini, metrikleri ve küçük çıktı dosyalarını işler. Böylece ham verinin pandas `groupby` ile belleğe alınması engellenir.

## Kurulum ve ön kontroller

Komutları depo kök dizininde çalıştırın.

```powershell
uv sync
```

Tahminden önce `analysis.duckdb`, `models/` altındaki gerekli model dosyaları ve `results/catboost_v4_2_hybrid_rule.json` mevcut olmalıdır. Tahmin sistemi veritabanını yalnızca okur; model eğitmez, kayıtlı dosyaları değiştirmez ve veritabanına kalıcı tablo ya da görünüm yazmaz.

## Veri yapısı ve temizleme işlemleri

`analysis.duckdb` projenin ana veritabanıdır. Modelin kullandığı `model_data_base` tablosunda sefer kimliği, kalkış tarihi ve saati, firma ve güzergâh kimlikleri, kanonik güzergâh kimliği, hedef değişken, takvim alanları, gün içindeki dakika ve 30 dakikalık kalkış dilimi bulunur. Hedef değişken `SEFER_SAYISI` alanından gelir ve temizlenmiş veride 1–300 aralığındadır.

Veri hazırlanırken üç temel önlem uygulanmıştır:

- 300'ün üzerindeki şüpheli kayıtlar kırpılmak yerine veri hatası kabul edilerek dışarıda bırakılmıştır;
- firma güzergâhı ile kanonik güzergâh eşleşmesi bulunmayan seferler çıkarılmıştır;
- sıralı durak listeleri oluşturulmadan önce yinelenen güzergâh-durak kayıtları temizlenmiştir.

Ham durak satırları doğrudan sefer tablosuna bağlanmamalıdır. Böyle bir bağlantı, her seferi durak sayısı kadar çoğaltır; sayım, ortalama ve hedef değerlerini bozar.

Temizlenmiş veri kümesinin yıllara göre dağılımı şöyledir:

| Yıl | Kayıt sayısı | Hedef ortalaması | Hedef ortancası |
|---|---:|---:|---:|
| 2023 | 2.587.983 | 32,13 | 31 |
| 2024 | 2.797.931 | 31,17 | 29 |
| 2025 | 2.937.192 | 31,79 | 29 |
| 2026, 14 Nisan'a kadar | 780.865 | 30,81 | 28 |

## Tahmin çalıştırma

### Tek sefer

```powershell
uv run python inference/predict_trip.py `
  --firma-id 123 --guzergah-kodu 456 `
  --date 2026-08-01 --time 14:30
```

Eşleşen geçmiş kayıt sayılarını, normalize edilmiş anahtarları ve seçilen taban çizgisi kaynağını görmek için `--debug` ekleyin.

### CSV ile toplu tahmin

```powershell
uv run python inference/predict_trips_batch.py --input input.csv --output output.csv
```

Girdi CSV dosyası şu sütunları içermelidir:

```csv
FIRMA_ID,GUZERGAH_KODU,SEFER_TARIHI,SEFER_SAATI
123,456,2026-08-01,14:30
```

### Seçili seferlerin denetimi

```powershell
uv run python inference/check_trips.py --input input.csv --output results/check.csv
```

Bu çıktı, tahmine ek olarak yalnızca teklif tarihinden önceki rota, tam saat ve hafta günü+saat istatistiklerini; dokuz inceleme bayrağını ve `any_review_flag` alanını içerir.

## Üretim tahmin akışı

Her teklif için yalnızca önerilen sefer tarihinden **kesin olarak önceki** kayıtlar kullanılır. Hedef sızıntısını önleyen temel kural budur.

1. Girdi doğrulanır ve normalize edilir.
2. Firma güzergâhı kanonik fiziksel güzergâha eşlenir.
3. Geçmiş özellikler DuckDB üzerinde vektörleştirilmiş sorgularla üretilir.
4. Dondurulmuş v4.1 CatBoost regresyon modeli sayısal tahmin üretir.
5. Hafta günü taban çizgisi hesaplanır; yüksek talep kanıtı yeterliyse dondurulmuş v4.2 kuralı tahmini yukarı yönlü harmanlar.
6. v4.4 sınıflandırıcıları `>=10`, `>=20`, `>=30` ve `>=43` eşikleri için çalıştırılır.
7. Tutarsız eşik kararları düzeltilir, talep etiketi atanır ve güvenilirlik raporlanır.

v4.1 regresyon modeli **64 özellik** kullanır: 6 kategorik, 2 takvim/saat, 40 uzun dönem geçmiş ve 16 yakın geçmiş özelliği. v4.4 sınıflandırıcıları bunlara 8 dağılım özelliği ekler ve **72 özellik** kullanır. Dondurulmuş bir `.cbm` dosyası kullanılacaksa özellik adları, sırası, türleri, gruplama anahtarları ve geri-düşüş davranışı değiştirilemez.

Üretim ortamında gereken dosyalar şunlardır:

| Dosya | Görevi |
|---|---|
| `models/catboost_demand_model_v4_1_recent_mae_6000.cbm` | 64 özellik kullanan v4.1 talep modeli |
| `results/catboost_v4_2_hybrid_rule.json` | seçilmiş v4.2 birleştirme kuralı |
| `models/catboost_demand_model_v4_4_classifier_ge_10_class_weights_none.cbm` | talebin 10 veya üzeri olma olasılığı |
| `models/catboost_demand_model_v4_4_classifier_ge_20_class_weights_none.cbm` | talebin 20 veya üzeri olma olasılığı |
| `models/catboost_demand_model_v4_4_classifier_ge_30_class_weights_none.cbm` | talebin 30 veya üzeri olma olasılığı |
| `models/catboost_demand_model_v4_4_classifier_ge_43.cbm` | talebin 43 veya üzeri olma olasılığı |
| bunlara karşılık gelen `*_metadata.json` dosyaları | karar eşikleri ve özellik listesinin kaydı |

10, 20 ve 30 eşikleri için sınıf ağırlığı kullanılmayan modeller; 43 eşiği için sınıfları dengeleyen model seçilmiştir. Tahmin sistemi açılırken bütün dosyaların varlığı, özellik adlarının kayıtlı listeyle aynı olması ve model sırasının değişmemiş olması denetlenir.

### Özellik grupları

Uzun dönem geçmiş bilgisi beş düzeyde özetlenir:

1. firma + kanonik güzergâh + 30 dakikalık saat dilimi + haftanın günü;
2. firma + kanonik güzergâh + 30 dakikalık saat dilimi;
3. kanonik güzergâh + 30 dakikalık saat dilimi + haftanın günü;
4. kanonik güzergâh;
5. firma.

Her düzeyde ortalama, ortanca, örnek standart sapması, en yüksek değer, yüzde 90'lık değer, 60 ve 100 üzeri kayıt oranları ile kayıt sayısı hesaplanır. Yakın geçmiş özellikleri; firma-güzergâh-saat-hafta günü ve kanonik güzergâh-saat-hafta günü düzeylerinde 30, 60, 90 ve 180 günlük pencereler kullanır. Yalnızca sınıflandırıcılarda kullanılan sekiz ek dağılım özelliği ise yüzde 10, yüzde 25 ve yüzde 75'lik değerler ile 10 altı ve 10, 20, 30, 40 üzeri kayıt oranlarıdır.

Hafta günü taban tahmini sırasıyla firma-güzergâh-saat-hafta günü, kanonik güzergâh-saat-hafta günü, kanonik güzergâh ve genel geçmiş ortalamasına geri düşer.

## Modelin geliştirilme süreci

Çalışma, açıklanabilir bir tarihsel ortalamadan başlayıp regresyon ile sınıflandırmayı bir araya getiren mevcut sisteme ulaşmıştır:

| Sürüm | Yapılan değişiklik | Sonuç |
|---|---|---|
| Taban 1 | firma + güzergâh + kalkış saati | ilk karşılaştırma noktası |
| Taban 2 | haftanın günü eklendi | seçilen taban model |
| Taban 3 | ay bilgisi eklendi | gruplar fazla seyrekleştiği için elendi |
| CatBoost v1 | kimlik ve takvim alanları | taban modeli geçemedi |
| CatBoost v2 | geçmiş ortalamaları ve kayıt sayıları | ilk belirgin makine öğrenmesi iyileşmesi |
| CatBoost v3 | daha geniş dağılım istatistikleri | MAE iyileşti; ağır işlemler DuckDB'ye taşındı |
| CatBoost v4.1 | 30/60/90/180 günlük yakın geçmiş | üretimde kullanılan regresyon modeli |
| CatBoost v4.2 | yüksek talep kanıtı varsa v4.1 ile taban tahmini birleştirildi | seçilen sayısal tahmin |
| CatBoost v4.3 | sekiz ek dağılım özelliği | yüksek talep ölçütleri kötüleştiği için elendi |
| CatBoost v4.4 | dört bağımsız talep eşiği sınıflandırıcısı | üretimde kullanılan sınıflandırma katmanı |
| CatBoost v4.5 | dini bayram özellikleri | iyileşme sağlamadığı için kullanıma alınmadı |

Ana başarı ölçütü MAE'dir; çünkü tahminin ortalama olarak kaç yolcu saptığını doğrudan anlatır. Büyük hataları görünür kılmak için RMSE, belirli talep aralıklarındaki sistematik yüksek veya düşük tahminleri görmek için de yanlılık değeri izlenir.

## Sonuçlar ve sınırlamalar

Seçilen v4.2 hibrit modelin MAE / RMSE sonuçları:

| Değerlendirme dönemi | MAE | RMSE |
|---|---:|---:|
| 2025 H1 doğrulama | 9,8537 | 14,5903 |
| 2025 H2 test | 9,8704 | 16,9336 |
| 2026 nihai değerlendirme | 9,7391 | 13,9263 |

2026 dönemi resmi nihai değerlendirmede bir kez kullanılmıştır. Dondurulmuş sonucu yeniden üretmek için tekrar çalıştırılabilir; değiştirilmiş bir modeli seçmek için kullanılmamalıdır.

Model yaygın talep aralığında daha güçlü, nadir yüksek talepli seferlerde ise düşük tahmine eğilimlidir. Açık tatil/etkinlik, rekabet, kapasite, fiyat, iptal ve operasyon maliyeti özellikleri yoktur. Bu nedenle nokta tahminini `prediction_reliability`, geçmiş kayıt sayıları, p90 benzeri bağlam ve eşik olasılıklarıyla birlikte değerlendirin.

## Araştırmayı yeniden çalıştırma ve yeni deney eğitimi

`scripts/` dizini eğitim ve değerlendirme akışlarını içerir. Her sürüm deneyi sabitlerini ve yardımcı kodlarını `scripts/shared/` dizininden alır; diğer sürümlerden bağımsızdır. Her betiği proje kökünden çalıştırın; dosyaları taşımayın ve göreli yolları değiştirmeyin.

```powershell
# Seçilen tarihsel ortalama taban çizgisi
uv run python scripts/baseline/time_dayofweek.py

# Dondurulmuş v4.1 regresyon modelinin eğitim akışı
uv run python scripts/catboost/v4_1/train.py

# 2025 H1 üzerinde v4.2 hibrit kural seçimi
uv run python scripts/catboost/v4_2/validation.py

# v4.4 eşik sınıflandırıcılarının eğitimi
uv run python scripts/catboost/v4_4/train.py

# Kronolojik değerlendirme örnekleri
uv run python scripts/catboost/v4_1/test.py
uv run python scripts/catboost/v4_2/test.py
uv run python scripts/catboost/v4_4/test.py
```

Bu işler yüksek işlem gücü gerektirebilir; DuckDB'nin geçici dosyaları için yeterli disk alanı ayrılmalıdır. GPU eğitimi CUDA desteği ister. Eğitim betikleri ilgili deneyin model ve sonuç yollarına yazar. Üretimde kullanılan kayıtlı dosyaların üzerine yazmayın; yeni çalışmalar için yeni dosya adı ve özellik sürümü kullanın.

Tüm değerlendirmeler kronolojik kalmalıdır. Yerleşik tasarımda 2024 denetimli eğitim satırları 2023 geçmişini kullanır; 2025 H1 doğrulaması 2023–2024 geçmişini, 2025 H2 testi 2025-06-30'a kadarki geçmişi ve resmi 2026 dönemi 2025-12-31'e kadarki geçmişi kullanır.

| Kullanım | Tarih aralığı | Kayıt sayısı |
|---|---|---:|
| Geniş eğitim geçmişi | 2023-01-01 – 2024-12-31 | 5.385.914 |
| Doğrulama | 2025-01-01 – 2025-06-30 | 1.420.182 |
| Test | 2025-07-01 – 2025-12-31 | 1.517.010 |
| Resmî nihai değerlendirme | 2026-01-01 – 2026-04-14 | 780.865 |

v4.1 modelinin gerçek denetimli eğitim matrisi, 2023 geçmişinden üretilen özelliklere sahip 2024 satırlarından oluşur. 2023 kayıtlarının doğrudan eğitim satırı olarak kullanılmamasının nedeni, onlar için aynı yapıda daha eski bir geçmiş bulunmamasıdır.

## API ve web arayüzü

Arka yüzü başlatın:

```powershell
npm run backend
```

OpenAPI arayüzü `http://localhost:8000/docs` adresindedir. Ortak API;
talep tahmini için `POST /predict`, tam durak ekleme değerlendirmesi için
`POST /predict-stop-addition`, `GET /health`, `GET /durak` altında durak
sorguları ve `GET /route` altında firma güzergâh sorguları sunar.

Başka bir terminalde arayüzü çalıştırın:

```powershell
npm install
npm run frontend
```

Vite, `/api` isteklerini yerel FastAPI sunucusuna yönlendirir. Üretim arayüz paketini `inference/frontend/dist/` altında oluşturmak için `npm run frontend:build` kullanın.

## Gelecekteki değişiklikler için kurallar

- Büyük tarama, birleştirme, gruplama ve yüzdelik hesaplarını DuckDB'de yapın; pandas'a yalnızca tamamlanmış model matrisini alın.
- Tarih aralıklarında bitiş gününü hariç tutan sorgular kullanın ve yalnızca tahmin döneminden önceki geçmişe başvurun.
- Zamana bağlı bu problemde rastgele eğitim/test ayrımı kullanmayın.
- Resmî 2026 dönemi üzerinde model ayarı yapmayın.
- Üretimde kullanılan model, eşik ve birleştirme kuralı dosyalarının üzerine yazmayın.
- Özellik adı, sırası, türü, gruplaması, istatistik tanımı veya eksik değer davranışındaki her değişikliği yeni bir özellik sürümü olarak ele alın.
- Tahminden önce `model.feature_names_` ile sıralı özellik listesinin birebir aynı olduğunu doğrulayın.
- Sonucu yolcu talebi olarak sunmadan önce kaynak sistemdeki `SEFER_SAYISI` alanının iş anlamını kesinleştirin.
