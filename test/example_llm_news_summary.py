from llm import GroqClient


client = GroqClient()


model_preference_list = ['llama-3.1-70b-versatile', 'mixtral-8x7b-32768', 'llama3-70b-8192']

active_models = client.get_active_models()
print(f'-> found active models: {active_models}')

model_name = None
for name in model_preference_list:
    if name in active_models:
        print(f'-> using {name}')
        model_name = name
        break

if model_name is None:
    raise ValueError('no preferred model is available')

NEWS_FEED = '''
Yenidoğan Çetesi davası | Çete üyeleri ilk kez hakim karşısında: 47 sanık için hesap zamanı!
Yenidoğan Çetesi davası başladı. Para uğruna bebeklerin yoğun bakımda ölümlerine neden olmakla suçlanan çete üyeleri ilk kez hakim karşısına çıktı. Davada 22’si tutuklu 47 sanık bulunuyor. Sanıkların 17 bin yıl hapisle cezalandırılmaları isteniyor. Çete üyelerinin yanı sıra, dosyada 19 özel hastane de sorumlu olarak yer alıyor. Dava duruşma salonunda yaşanan gerginlik nedeniyle gecikmeli başladı. Mahkeme heyeti davada ilk olarak sanıkların kimlik tespitlerini yaptı. İddianamede elebaşı olarak tanımlanan Fırat Sarı, aylık gelirinin 400 bin lira olduğunu söyledi. Kimlik tespiti yaklaşık 2,5 saat sürdü. Savcılık makamı, doğrudan mağdur olmayanların müdahillik talebinin reddini isterken duruşmaya kısa bir ara verildi. Muhabir Melike Şahin, duruşma salonundan son gelişmeleri aktarıyor.

Uzaklaştırma kararı aldırmaya giderken eşi tarafından vuruldu
Manisa'nın Şehzadeler ilçesinde Belgin B. uzaklaştırma kararı aldırmak için adliyeye giderken boşanma aşamasındaki eşi Furkan B. tarafından tabancayla baldırından vuruldu.Nurlupınar Mahallesi 308 Sokak'ta Belgin B., mart ayında boşanma davası açtığı eşi Furkan B. için uzaklaştırma kararı aldırmak için 3 aylık kızıyla adliyeye gitmek için evden çıktı.   Bebek arabasındaki kızıyla yolda yürüyen Belgin B., bu sırada motosikletiyle kendisini takip eden eşi Furkan B. tarafından sağ baldırından tabancayla yaralandı.   Furkan B. motosikletiyle kaçtı. SAĞLIK DURUMLARI İYİ  Çevredekiler durumu sağlık ekipleri ve polise bildirdi.  Sağlık ekibinin olay yerindeki ilk müdahalesinin ardından Manisa Şehir Hastanesi'ne kaldırılan Belgin B., tedaviye alındı. Belgin B. ve yanındaki 3 aylık kızının sağlık durumlarının iyi olduğu öğrenildi.  Polis, olayın ardından kaçan Furkan B.'yi yakaladı. Gözaltına alınan Furkan B., polisteki işlemlerinin ardından sevk edildiği adliyede tutuklandı.

Zonguldak'ta sis etkili oldu
Zonguldak'ta hava sıcaklıklarının 11 dereceye kadar düşmesiyle birlikte hem şehrin yüksek kesimlerinde hem de deniz trafiğinde sis etkili oldu.Zonguldak’ta bugün sabah saatlerinden sis ve yağmur etkisini gösterdi. Özellikle şehir merkezi ve yüksek kesimlerde görüş mesafesi azalırken, deniz trafiğinde de sis etkili oldu.  Liman dışında bazı yük gemileri sıra beklerken, balıkçı tekneleri sisin içinde ilerlemeye çalıştı. Ereğli ilçesi yönüne giden kara yolunda, sis nedeniyle görüş mesafesi yer yer 20-30 metreye kadar düştü.  Meteoroloji Genel Müdürlüğü, Zonguldak'ta yağışın etkisini hafta sonuna kadar aralıklarla sürdüreceğini açıkladı.
'''

print()
print('--- FEED START ---')
print(NEWS_FEED)
print('--- FEED END ---')
print()

print(f'-> summarizing using {model_name}\n')

content = client.get_summary(NEWS_FEED, model_name)
print(f'--- SUMMARY START ---\n')
summaries = []
for idx in range(content.num_summaries):
    summaries.append(f'{content.headers[idx]}\n{content.summaries[idx]}\n')
print('\n'.join(summaries))
print(f'--- SUMMARY END ---')
