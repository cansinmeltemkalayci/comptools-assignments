# Adımlar

## 1. Ortam Kurulumu

[+] Sanal ortam oluşturuldu.
```shell
python3 -m venv venv
```

[+] Gerekli kütüphaneler için requirements.txt hazırlandı.

[+] Sanal ortam aktifleştirildi.
```shell
source venv/bin/activate
```

[+] Gerekli kütüphaneler yüklendi.
```shell
pip install -r requirements.txt
```

## 2. Veri Yükleme ve Hazırlama

[+] FRED-MD veri dosyası indirildi.
```
https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv
```

[+] Veri dosyası pandas ile okundu.

[+] Dönüşüm kodları satırı ayrıştırıldı ve DataFrame'den çıkarıldı.

[+] 'sasdate' sütunu datetime formatına dönüştürüldü.

[+] Eksik tarih verileri olan satırlar silindi.

## 3. Veri Dönüşümleri

[+] Dönüşüm kodları bir veri yapısına kaydedildi.

[+] apply_transformation() fonksiyonu tanımlandı:
  - 1: Dönüşüm yok
  - 2: Birinci fark (Δx_t)
  - 3: İkinci fark (Δ²x_t)
  - 4: Logaritma (log(x_t))
  - 5: Logaritmanın birinci farkı (Δlog(x_t))
  - 6: Logaritmanın ikinci farkı (Δ²log(x_t))
  - 7: Yüzde değişim (x_t/x_{t-1} - 1)

[+] Tüm değişkenlere uygun dönüşümler uygulandı.

[+] Dönüşümler nedeniyle oluşan eksik değerlere sahip ilk iki satır silindi.

## 4. Veri Keşfi ve Görselleştirme

[+] Her değişkendeki eksik değer yüzdesi hesaplandı.

[+] Model 1 için değişkenler seçildi: INDPRO, CPIAUCSL, TB3MS.
  - INDPRO: Sanayi Üretimi
  - CPIAUCSL: Enflasyon (CPI)
  - TB3MS: 3 aylık Hazine Bonosu faiz oranı

[+] Model 2 için değişkenler seçildi: INDPRO, ACOGNO, BUSLOANS.
  - INDPRO: Sanayi Üretimi
  - ACOGNO: Tüketim Malları Üreticilerinin Yeni Siparişleri
  - BUSLOANS: Ticari ve Endüstriyel Krediler

[+] Model 1 ve Model 2 için dönüştürülmüş değişkenlerin zaman serisi grafikleri oluşturuldu.

## 5. Tahmin Modelleri

[+] ARX modelini uygulamak için calculate_forecast() fonksiyonu geliştirildi.

[+] Fonksiyon şu işlemleri gerçekleştirir:
  - Belirli bir tarihe kadar veriyi filtreler
  - Belirli ufuklar için gerçek değerleri alır
  - Bağımlı değişkenin ve bağımsız değişkenlerin gecikmeli değerlerini içeren tasarım matrisi oluşturur
  - OLS ile model parametrelerini tahmin eder
  - Farklı ufuklar için tahminleri ve tahmin hatalarını hesaplar

## 6. Gerçek Zamanlı Değerlendirme

[+] 1999-12-01 ile 2005-12-01 arasındaki dönem için değerlendirme tarihleri oluşturuldu.

[+] Her değerlendirme tarihi için:
  - Model 1 ve Model 2 kullanılarak 1, 4 ve 8 ay ilerisi için tahminler oluşturuldu
  - Tahmin hataları hesaplandı
  - Sonuçlar listeler halinde kaydedildi

[+] Liste verileri DataFrame'lere dönüştürüldü.

[+] Her tahmin ufku ve model için MSFE ve RMSFE hesaplandı.

## 7. Sonuçların Görselleştirilmesi

[+] Model 1 ve Model 2'nin 1 aylık tahminleri ile gerçek değerlerin karşılaştırma grafiği oluşturuldu.

[+] Tahmin hatalarının tüm ufuklar için zaman içindeki değişimini gösteren grafikler oluşturuldu.

[+] İki modelin RMSFE değerlerini karşılaştıran çubuk grafik oluşturuldu.

## 8. Farklı Model Yapılarının Karşılaştırılması

[+] Farklı gecikme yapıları (1, 2, 4, 6, 12) için her iki model ve her tahmin ufkunda RMSFE değerleri hesaplandı.

[+] Farklı gecikme yapılarının performansını gösteren grafikler oluşturuldu.

[+] Her tahmin ufku için en iyi gecikme yapısı belirlendi.

## 9. Sonuçların Kaydedilmesi ve Raporlanması

[+] Tüm sonuçlar ve grafikler outputs klasörüne kaydedildi.

[+] Sonuçlar çıktı olarak rapor edildi:
  - Her modelin RMSFE değerleri
  - Model karşılaştırmaları (hangi model daha iyi, ne kadar daha iyi)
  - Her model için en iyi gecikme yapıları

## 10. Sonuçlar ve Çıkarımlar

[+] Model 1 (CPIAUCSL, TB3MS) tüm tahmin ufuklarında Model 2'den (ACOGNO, BUSLOANS) daha iyi performans gösterdi.

[+] En belirgin fark 8 aylık tahmin ufkunda görüldü (%14.51 daha iyi).

[+] Model 1 için en iyi gecikme yapıları:
  - 1 aylık ufuk için p=1
  - 4 aylık ufuk için p=12
  - 8 aylık ufuk için p=6

[+] Model 2 için en iyi gecikme yapıları:
  - 1 aylık ufuk için p=4
  - 4 aylık ufuk için p=1
  - 8 aylık ufuk için p=1 