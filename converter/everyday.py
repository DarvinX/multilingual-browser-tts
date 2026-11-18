"""Author held-out everyday Roman/native examples without using converter outputs."""

import json
from pathlib import Path


# Four greetings and twenty-five everyday requests, observations, and replies
# provide 100 distinct two-sentence inputs, including informal spellings.
# The spelling choices are author-written references and need native review.
PHRASES = {
    "bn": """কেমন আছো?|Kemon achho?
একটু শোনো।|Ektu shono.
একটা কথা বলি?|Ekta kotha boli?
আচ্ছা, ঠিক আছে।|Achha, thik achhe.
আজ বিকেলে চা খেতে যাবে?|Aaj bikele cha khete jabe?
তুমি কখন বাড়ি ফিরবে?|Tumi kokhon bari phirbe?
আমার ফোনটা কোথায় রেখেছি?|Amar phonta kothay rekhechi?
বাইরে বেশ বৃষ্টি হচ্ছে।|Baire besh brishti hochchhe.
দরজাটা একটু বন্ধ করো।|Dorjata ektu bondho koro.
তুমি কি খেয়েছো?|Tumi ki kheyechho?
আমাকে পরে ফোন করো।|Amake pore phone koro.
আমার একটু জল চাই।|Amar ektu jol chai.
কাল সকালে দেখা হবে।|Kaal shokale dekha hobe.
আমাদের জন্য একটু অপেক্ষা করো।|Amader jonno ektu opekkha koro.
তুমি কি বাজারে যাচ্ছো?|Tumi ki bajare jachchho?
বাসটা কখন আসবে?|Basta kokhon ashbe?
আমি চাবিটা খুঁজে পাচ্ছি না।|Ami chabita khuje pachchhi na.
আজ তোমার কাজ কেমন হলো?|Aaj tomar kaj kemon holo?
চলো একটু হাঁটতে যাই।|Cholo ektu hatte jai.
আমার খুব ঘুম পাচ্ছে।|Amar khub ghum pachchhe.
তুমি কি আমাকে সাহায্য করবে?|Tumi ki amake shahajjo korbe?
কাল একটা ছাতা নিয়ে এসো।|Kaal ekta chhata niye esho.
বইটা তোমার টেবিলে আছে।|Boita tomar tebile achhe.
এখানে বসলে ভালো লাগবে।|Ekhane boshle bhalo lagbe.
এটা বানাতে কতক্ষণ লাগবে?|Eta banate kotokkhon lagbe?
আরেকটু ধীরে বলবে?|Arektu dhire bolbe?
আমরা একসঙ্গে রান্না করব।|Amra ekshonge ranna korbo.
বাচ্চারা আজ তাড়াতাড়ি ঘুমিয়েছে।|Bachchara aaj taratari ghumiyechhe.
বাড়ি পৌঁছে খবর দিও।|Bari pounchhe khobor dio.""",
    "gu": """કેમ છો?|Kem chho?
જરા સાંભળો.|Jara sambhalo.
એક વાત પૂછું?|Ek vaat puchhu?
હા, બરાબર છે.|Haa, barabar chhe.
આજે સાંજે ચા પીવા આવશો?|Aaje saanje cha piva aavsho?
તમે ઘરે ક્યારે આવશો?|Tame ghare kyare aavsho?
મેં ફોન ક્યાં મૂક્યો છે?|Me phone kya mukyo chhe?
બહાર ઘણો વરસાદ પડે છે.|Bahar ghano varsad pade chhe.
બારણું જરા બંધ કરો.|Baranu jara bandh karo.
તમે જમી લીધું?|Tame jami lidhu?
મને પછી ફોન કરજો.|Mane pachhi phone karjo.
મને થોડું પાણી આપશો?|Mane thodu paani aapsho?
કાલે સવારે મળીએ.|Kaale savare maliye.
અમારી થોડી રાહ જુઓ.|Amari thodi raah juo.
તમે બજારમાં જાઓ છો?|Tame bajarma jao chho?
બસ ક્યારે આવશે?|Bas kyare aavshe?
મને ચાવી મળતી નથી.|Mane chavi malti nathi.
આજે તમારું કામ કેવું રહ્યું?|Aaje tamaru kaam kevu rahyu?
ચાલો થોડું ચાલવા જઈએ.|Chalo thodu chalva jaiye.
મને બહુ ઊંઘ આવે છે.|Mane bahu ungh aave chhe.
તમે મને મદદ કરશો?|Tame mane madad karsho?
કાલે એક છત્રી લઈ આવજો.|Kaale ek chhatri lai aavjo.
પુસ્તક તમારા ટેબલ પર છે.|Pustak tamara tebal par chhe.
અહીં બેસીએ તો સારું રહેશે.|Ahin besiye to saru raheshe.
આ બનાવવામાં કેટલો સમય લાગશે?|Aa banavvama ketlo samay lagshe?
જરા ધીમેથી બોલશો?|Jara dhime thi bolsho?
આપણે સાથે રસોઈ કરીશું.|Aapne saathe rasoi karishu.
બાળકો આજે વહેલા સૂઈ ગયાં.|Balko aaje vahela sui gaya.
ઘરે પહોંચીને મને કહેજો.|Ghare pahochine mane kahejo.""",
    "hi": """कैसे हो?|Kaise ho?
ज़रा सुनो।|Zara suno.
एक बात पूछूँ?|Ek baat poochhun?
हाँ, ठीक है।|Haan, theek hai.
आज शाम चाय पीने चलोगे?|Aaj shaam chai peene chaloge?
तुम घर कब लौटोगे?|Tum ghar kab lautoge?
मैंने फ़ोन कहाँ रखा है?|Maine phone kahan rakha hai?
बाहर बहुत बारिश हो रही है।|Bahar bahut baarish ho rahi hai.
दरवाज़ा ज़रा बंद कर दो।|Darwaza zara band kar do.
तुमने खाना खा लिया?|Tumne khana kha liya?
मुझे बाद में फ़ोन करना।|Mujhe baad mein phone karna.
मुझे थोड़ा पानी चाहिए।|Mujhe thoda paani chahiye.
कल सुबह मिलते हैं।|Kal subah milte hain.
हमारा थोड़ा इंतज़ार करना।|Hamara thoda intezar karna.
क्या तुम बाज़ार जा रहे हो?|Kya tum bazaar ja rahe ho?
बस कब आएगी?|Bus kab aayegi?
मुझे चाबी नहीं मिल रही।|Mujhe chaabi nahi mil rahi.
आज तुम्हारा काम कैसा रहा?|Aaj tumhara kaam kaisa raha?
चलो थोड़ी देर टहलते हैं।|Chalo thodi der tahalte hain.
मुझे बहुत नींद आ रही है।|Mujhe bahut neend aa rahi hai.
क्या तुम मेरी मदद करोगे?|Kya tum meri madad karoge?
कल एक छाता ले आना।|Kal ek chhata le aana.
किताब तुम्हारी मेज़ पर है।|Kitaab tumhari mez par hai.
यहाँ बैठना अच्छा रहेगा।|Yahan baithna achha rahega.
इसे बनाने में कितना समय लगेगा?|Ise banane mein kitna samay lagega?
ज़रा धीरे बोलोगे?|Zara dheere bologe?
हम साथ मिलकर खाना बनाएँगे।|Hum saath milkar khana banayenge.
बच्चे आज जल्दी सो गए।|Bacche aaj jaldi so gaye.
घर पहुँचकर ख़बर देना।|Ghar pahunchkar khabar dena.""",
    "kn": """ಹೇಗಿದ್ದೀಯ?|Hegiddiya?
ಸ್ವಲ್ಪ ಕೇಳು.|Swalpa kelu.
ಒಂದು ವಿಷಯ ಕೇಳಲಾ?|Ondu vishaya kelala?
ಹೌದು, ಸರಿ.|Haudu, sari.
ಇಂದು ಸಂಜೆ ಚಹಾ ಕುಡಿಯಲು ಬರುತ್ತೀಯಾ?|Indu sanje chaha kudiyalu baruttiya?
ನೀನು ಮನೆಗೆ ಯಾವಾಗ ಬರುತ್ತೀಯಾ?|Neenu manege yavaga baruttiya?
ನನ್ನ ಫೋನ್ ಎಲ್ಲಿ ಇಟ್ಟಿದ್ದೇನೆ?|Nanna phone elli ittiddene?
ಹೊರಗೆ ಜೋರಾಗಿ ಮಳೆ ಬರುತ್ತಿದೆ.|Horage joragi male baruttide.
ಬಾಗಿಲು ಸ್ವಲ್ಪ ಮುಚ್ಚು.|Bagilu swalpa muchchu.
ನೀನು ಊಟ ಮಾಡಿದೆಯಾ?|Neenu oota madideya?
ನನಗೆ ಆಮೇಲೆ ಫೋನ್ ಮಾಡು.|Nanage aamele phone madu.
ನನಗೆ ಸ್ವಲ್ಪ ನೀರು ಬೇಕು.|Nanage swalpa neeru beku.
ನಾಳೆ ಬೆಳಗ್ಗೆ ಭೇಟಿಯಾಗೋಣ.|Naale belagge bhetiyagona.
ನಮಗಾಗಿ ಸ್ವಲ್ಪ ಕಾಯು.|Namagagi swalpa kayu.
ನೀನು ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗುತ್ತಿದ್ದೀಯಾ?|Neenu marukattege hoguttiddiya?
ಬಸ್ಸು ಯಾವಾಗ ಬರುತ್ತದೆ?|Bassu yavaga baruttade?
ನನಗೆ ಕೀಲಿ ಸಿಗುತ್ತಿಲ್ಲ.|Nanage keeli siguttilla.
ಇಂದು ನಿನ್ನ ಕೆಲಸ ಹೇಗಿತ್ತು?|Indu ninna kelasa hegittu?
ಬಾ, ಸ್ವಲ್ಪ ನಡೆಯೋಣ.|Baa, swalpa nadeyona.
ನನಗೆ ತುಂಬಾ ನಿದ್ದೆ ಬರುತ್ತಿದೆ.|Nanage tumba nidde baruttide.
ನನಗೆ ಸಹಾಯ ಮಾಡುತ್ತೀಯಾ?|Nanage sahaya maduttiya?
ನಾಳೆ ಒಂದು ಛತ್ರಿ ತಗೊಂಡು ಬಾ.|Naale ondu chatri tagondu baa.
ಪುಸ್ತಕ ನಿನ್ನ ಮೇಜಿನ ಮೇಲಿದೆ.|Pustaka ninna mejina melide.
ಇಲ್ಲಿ ಕುಳಿತರೆ ಚೆನ್ನಾಗಿರುತ್ತದೆ.|Illi kulitare chennagiruttade.
ಇದನ್ನು ಮಾಡಲು ಎಷ್ಟು ಸಮಯ ಬೇಕು?|Idannu madalu eshtu samaya beku?
ಸ್ವಲ್ಪ ನಿಧಾನವಾಗಿ ಹೇಳುತ್ತೀಯಾ?|Swalpa nidhanavagi heluttiya?
ನಾವು ಒಟ್ಟಿಗೆ ಅಡುಗೆ ಮಾಡೋಣ.|Naavu ottige aduge madona.
ಮಕ್ಕಳು ಇಂದು ಬೇಗ ಮಲಗಿದರು.|Makkalu indu bega malagidaru.
ಮನೆ ತಲುಪಿದ ಮೇಲೆ ತಿಳಿಸು.|Mane talupida mele tilisu.""",
    "ml": """സുഖമാണോ?|Sukhamaano?
ഒന്ന് കേൾക്കൂ.|Onnu kelkku.
ഒരു കാര്യം ചോദിക്കട്ടെ?|Oru kaaryam chodikkatte?
അതെ, ശരിയാണ്.|Athe, shariyaanu.
ഇന്ന് വൈകിട്ട് ചായ കുടിക്കാൻ വരുമോ?|Innu vaikittu chaaya kudikkaan varumo?
നീ വീട്ടിലേക്ക് എപ്പോഴാണ് വരുന്നത്?|Nee veettilekku eppozhaanu varunnathu?
ഞാൻ ഫോൺ എവിടെയാണ് വെച്ചത്?|Njaan phone evideyaanu vechathu?
പുറത്ത് നല്ല മഴയുണ്ട്.|Purathu nalla mazhayundu.
വാതിൽ ഒന്ന് അടയ്ക്കൂ.|Vaathil onnu adaykku.
നീ ഭക്ഷണം കഴിച്ചോ?|Nee bhakshanam kazhicho?
എന്നെ പിന്നെ വിളിക്കൂ.|Enne pinne vilikku.
എനിക്ക് കുറച്ച് വെള്ളം വേണം.|Enikku kurachu vellam venam.
നാളെ രാവിലെ കാണാം.|Naale raavile kaanaam.
ഞങ്ങൾക്കായി കുറച്ച് കാത്തിരിക്കൂ.|Njangalkaayi kurachu kaathirikku.
നീ ചന്തയിൽ പോകുന്നുണ്ടോ?|Nee chanthayil pokunnundo?
ബസ് എപ്പോൾ വരും?|Bus eppol varum?
എനിക്ക് താക്കോൽ കിട്ടുന്നില്ല.|Enikku thaakkol kittunnilla.
ഇന്ന് നിന്റെ ജോലി എങ്ങനെയുണ്ടായിരുന്നു?|Innu ninte joli enganeyundaayirunnu?
വാ, നമുക്ക് കുറച്ച് നടക്കാം.|Vaa, namukku kurachu nadakkaam.
എനിക്ക് നല്ല ഉറക്കം വരുന്നു.|Enikku nalla urakkam varunnu.
നീ എന്നെ സഹായിക്കുമോ?|Nee enne sahaayikkumo?
നാളെ ഒരു കുട കൊണ്ടുവരൂ.|Naale oru kuda konduvaru.
പുസ്തകം നിന്റെ മേശപ്പുറത്തുണ്ട്.|Pusthakam ninte meshappurathundu.
ഇവിടെ ഇരുന്നാൽ നന്നായിരിക്കും.|Ivide irunnaal nannaayirikkum.
ഇത് ഉണ്ടാക്കാൻ എത്ര സമയം വേണം?|Ithu undaakkaan ethra samayam venam?
കുറച്ച് പതുക്കെ പറയുമോ?|Kurachu pathukke parayumo?
നമുക്ക് ഒരുമിച്ച് പാചകം ചെയ്യാം.|Namukku orumichu paachakam cheyyaam.
കുട്ടികൾ ഇന്ന് നേരത്തെ ഉറങ്ങി.|Kuttikal innu nerathe urangi.
വീട്ടിൽ എത്തിയിട്ട് അറിയിക്കൂ.|Veettil ethiyittu ariyikku.""",
    "ta": """எப்படி இருக்கிறாய்?|Eppadi irukkirai?
கொஞ்சம் கேளு.|Konjam kelu.
ஒரு விஷயம் கேட்கலாமா?|Oru vishayam ketkalama?
ஆமாம், சரிதான்.|Aamaam, sarithaan.
இன்று மாலை தேநீர் குடிக்க வருவாயா?|Indru maalai theneer kudikka varuvaaya?
நீ எப்போது வீட்டுக்கு வருவாய்?|Nee eppodhu veettukku varuvaai?
என் போனை எங்கே வைத்தேன்?|En phoneai enge vaithen?
வெளியே நன்றாக மழை பெய்கிறது.|Veliye nandraga mazhai peigiradhu.
கதவைக் கொஞ்சம் மூடு.|Kadhavai konjam moodu.
நீ சாப்பிட்டாயா?|Nee saappittaaya?
என்னைப் பிறகு கூப்பிடு.|Ennai piragu koopidu.
எனக்குக் கொஞ்சம் தண்ணீர் வேண்டும்.|Enakku konjam thanneer vendum.
நாளை காலையில் பார்க்கலாம்.|Naalai kaalaiyil paarkkalaam.
எங்களுக்காகக் கொஞ்சம் காத்திரு.|Engalukkaaga konjam kaathiru.
நீ சந்தைக்குப் போகிறாயா?|Nee sandhaikku pogiraaya?
பேருந்து எப்போது வரும்?|Perundhu eppodhu varum?
எனக்குச் சாவி கிடைக்கவில்லை.|Enakku saavi kidaikkavillai.
இன்று உன் வேலை எப்படி இருந்தது?|Indru un velai eppadi irundhadhu?
வா, கொஞ்சம் நடக்கலாம்.|Vaa, konjam nadakkalaam.
எனக்கு ரொம்பத் தூக்கம் வருகிறது.|Enakku romba thookkam varugiradhu.
நீ எனக்கு உதவி செய்வாயா?|Nee enakku udhavi seivaaya?
நாளை ஒரு குடை கொண்டுவா.|Naalai oru kudai konduvaa.
புத்தகம் உன் மேஜையில் இருக்கிறது.|Puthagam un mejaiyil irukkiradhu.
இங்கே உட்கார்ந்தால் நன்றாக இருக்கும்.|Inge utkaarndhaal nandraga irukkum.
இதைச் செய்ய எவ்வளவு நேரம் ஆகும்?|Idhai seyya evvalavu neram aagum?
கொஞ்சம் மெதுவாகச் சொல்வாயா?|Konjam medhuvaaga solvaaya?
நாம் சேர்ந்து சமைக்கலாம்.|Naam serndhu samaikkalaam.
குழந்தைகள் இன்று சீக்கிரம் தூங்கிவிட்டார்கள்.|Kuzhandhaigal indru seekiram thoongivittaargal.
வீட்டுக்குப் போனதும் தகவல் சொல்லு.|Veettukku ponadhum thagaval sollu.""",
    "te": """ఎలా ఉన్నావు?|Ela unnava?
కొంచెం విను.|Konchem vinu.
ఒక విషయం అడగనా?|Oka vishayam adaganaa?
అవును, సరే.|Avunu, sare.
ఈ రోజు సాయంత్రం టీ తాగడానికి వస్తావా?|Ee roju saayantram tea taagadaaniki vastaavaa?
నువ్వు ఇంటికి ఎప్పుడు వస్తావు?|Nuvvu intiki eppudu vastaavu?
నా ఫోన్ ఎక్కడ పెట్టాను?|Naa phone ekkada pettaanu?
బయట బాగా వాన పడుతోంది.|Bayata baagaa vaana padutondi.
తలుపు కొంచెం మూసేయి.|Talupu konchem mooseyi.
నువ్వు భోజనం చేశావా?|Nuvvu bhojanam chesaavaa?
నాకు తర్వాత ఫోన్ చేయి.|Naaku tarvata phone cheyi.
నాకు కొంచెం నీళ్లు కావాలి.|Naaku konchem neellu kaavali.
రేపు ఉదయం కలుద్దాం.|Repu udayam kaluddaam.
మా కోసం కొంచెం ఆగు.|Maa kosam konchem aagu.
నువ్వు మార్కెట్‌కు వెళ్తున్నావా?|Nuvvu marketku veltunnaavaa?
బస్సు ఎప్పుడు వస్తుంది?|Bassu eppudu vastundi?
నాకు తాళం చెవి దొరకడం లేదు.|Naaku taalam chevi dorakadam ledu.
ఈ రోజు నీ పని ఎలా జరిగింది?|Ee roju nee pani ela jarigindi?
రా, కొంచెం నడుద్దాం.|Raa, konchem naduddaam.
నాకు చాలా నిద్ర వస్తోంది.|Naaku chaala nidra vastondi.
నువ్వు నాకు సహాయం చేస్తావా?|Nuvvu naaku sahaayam chestaavaa?
రేపు ఒక గొడుగు తీసుకురా.|Repu oka godugu teesukuraa.
పుస్తకం నీ బల్ల మీద ఉంది.|Pustakam nee balla meeda undi.
ఇక్కడ కూర్చుంటే బాగుంటుంది.|Ikkada koorchunte baaguntundi.
ఇది చేయడానికి ఎంత సమయం పడుతుంది?|Idi cheyadaaniki enta samayam padutundi?
కొంచెం మెల్లగా చెబుతావా?|Konchem mellaga chebutaavaa?
మనం కలిసి వంట చేద్దాం.|Manam kalisi vanta cheddaam.
పిల్లలు ఈ రోజు త్వరగా పడుకున్నారు.|Pillalu ee roju tvaraga padukunnaaru.
ఇంటికి చేరాక చెప్పు.|Intiki cheraaka cheppu.""",
    "en": """How are you doing?
Listen for a second.
Can I ask you something?
All right, that works.
Do you want to get tea this evening?
When will you be home?
Where did I put my phone?
It's raining pretty hard outside.
Could you close the door for me?
Have you eaten yet?
Give me a call later.
Could I have a little water?
Let's meet tomorrow morning.
Wait here for us a minute.
Are you heading to the market?
When is the bus coming?
I can't find my key.
How was work today?
Let's go for a short walk.
I'm starting to get sleepy.
Can you give me a hand?
Bring an umbrella tomorrow.
The book is on your desk.
This looks like a good place to sit.
How long will this take to make?
Could you say that a little slower?
We can cook dinner together.
The kids went to bed early today.
Let me know when you get home.""",
}


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    for language, lines in PHRASES.items():
        phrases = [tuple(line.split("|", 1)) if "|" in line else (line, line) for line in lines.splitlines()]
        assert len(phrases) == 29
        rows = [{"id": f"{language}-{i:03d}", "language": language, "native": f"{prefix[0]} {body[0]}",
                 "roman": f"{prefix[1]} {body[1]}"} for i, (prefix, body) in enumerate(
                     (prefix, body) for prefix in phrases[:4] for body in phrases[4:])]
        assert len({row["roman"] for row in rows}) == len({row["native"] for row in rows}) == 100
        (output / f"{language}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    build(Path(__file__).with_name("fixtures"))
