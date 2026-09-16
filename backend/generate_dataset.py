"""
generate_dataset.py
--------------------
Builds a synthetic-but-realistic Marathi news-headline dataset for the
12 categories used by the frontend (index.html / script.js):

Politics, Sports, Entertainment, Health, Crime, Tech, Travel,
Education, Fashion, Bhakti, Auto, International

Why synthetic data?
    A real, labelled, publicly-licensed Marathi news corpus large enough
    to train on was not available in this environment, and scraping /
    redistributing copyrighted news text would not be appropriate for a
    student project either. Instead this script combines category-specific
    Marathi vocabulary (the same words already used for the keyword
    simulation in the original script.js) with sentence templates and
    city/entity names to programmatically build several hundred distinct,
    grammatically reasonable headlines per category.

    This is enough for TF-IDF + Multinomial Naive Bayes to genuinely learn
    per-category word-frequency patterns (rather than being hard-coded),
    which is the point of the NLP mini-project. For a stronger real-world
    model, swap this CSV for a scraped/labelled corpus (e.g. from Lokmat,
    Loksatta, Saamana RSS feeds) using the same two columns:
    `headline,category`.
"""

import csv
import random

random.seed(42)

PLACES = [
    "मुंबई", "पुणे", "नागपूर", "नाशिक", "औरंगाबाद", "कोल्हापूर",
    "ठाणे", "सोलापूर", "सांगली", "अमरावती", "रत्नागिरी", "सातारा",
]

# category -> (keywords, templates)
# {kw} = category keyword, {place} = city name
CATEGORY_DATA = {
    "Politics": {
        "keywords": [
            "सरकार", "मुख्यमंत्री", "निवडणूक", "विधानसभा", "मंत्री", "पक्ष",
            "कायदा", "राजकारण", "पंतप्रधान", "संसद", "नेता", "मतदान",
            "राज्यपाल", "अर्थसंकल्प", "आमदार", "खासदार", "युती", "आघाडी",
        ],
        "templates": [
            "{place} मध्ये {kw} विषयी मोठी घोषणा",
            "{kw} संदर्भात राज्य सरकारचा नवा निर्णय",
            "{place} येथे {kw} बाबत जोरदार चर्चा",
            "{kw} च्या मुद्द्यावरून विरोधकांचा हल्लाबोल",
            "आगामी {kw} साठी उमेदवारांची यादी जाहीर",
            "{kw} बैठकीत महत्त्वाचे निर्णय घेण्यात आले",
            "{place} मधील {kw} बाबत जनतेमध्ये उत्सुकता",
            "{kw} बाबत नवीन विधेयक सादर",
        ],
    },
    "Sports": {
        "keywords": [
            "सामना", "क्रिकेट", "फुटबॉल", "कबड्डी", "टेनिस", "स्पर्धा",
            "चॅम्पियन", "गोल", "धाव", "मैदान", "खेळाडू", "विजय",
            "पराभव", "ऑलिंपिक", "वर्ल्डकप", "बॅडमिंटन", "सामनावीर",
        ],
        "templates": [
            "{place} येथे रंगणार आजचा {kw}",
            "भारतीय संघाचा {kw} स्पर्धेत दमदार {kw2}",
            "{place} च्या खेळाडूने {kw} मध्ये उत्तम कामगिरी",
            "{kw} सामन्यात संघाची जबरदस्त सुरुवात",
            "अंतिम {kw} जिंकत संघाने पटकावला किताब",
            "{place} स्टेडियमवर {kw} सामन्याची जय्यत तयारी",
            "युवा {kw} स्टारची निवड राष्ट्रीय संघात",
            "{kw} मध्ये विक्रमी कामगिरी करत खेळाडू चर्चेत",
        ],
    },
    "Entertainment": {
        "keywords": [
            "अभिनेता", "चित्रपट", "गाणी", "नाट्य", "सिनेमा", "कलाकार",
            "प्रदर्शन", "रिलीज", "अभिनेत्री", "दिग्दर्शक", "संगीत",
            "पुरस्कार", "मालिका", "बॉलिवूड",
        ],
        "templates": [
            "नवीन {kw} च्या {kw2} तारखेची घोषणा",
            "प्रसिद्ध {kw} ची पुढील {kw2} विषयी माहिती समोर",
            "{place} मध्ये {kw} च्या शूटिंगला सुरुवात",
            "{kw} ला प्रेक्षकांचा उदंड प्रतिसाद",
            "{kw} सोहळ्यात यंदा कोणाला मिळणार पुरस्कार",
            "लोकप्रिय {kw} ची नवी झलक प्रदर्शित",
            "{place} फिल्म फेस्टिव्हलमध्ये {kw} ची चर्चा",
        ],
    },
    "Health": {
        "keywords": [
            "डॉक्टर", "आरोग्य", "उपचार", "हॉस्पिटल", "दवाखाना", "रोग",
            "लस", "आजार", "रुग्ण", "शस्त्रक्रिया", "औषध", "संसर्ग", "तपासणी",
        ],
        "templates": [
            "{place} मध्ये {kw} शिबिराचे आयोजन",
            "हिवाळ्यात {kw} बाबत तज्ज्ञ {kw2} यांचा सल्ला",
            "{kw} वाढल्याने {place} मध्ये खबरदारीचे आवाहन",
            "नवीन {kw} मुळे रुग्णांना दिलासा",
            "{place} च्या हॉस्पिटलमध्ये मोफत {kw} शिबीर",
            "{kw} संदर्भात आरोग्य विभागाची मार्गदर्शक सूचना",
            "गंभीर {kw} नंतर रुग्णाची प्रकृती सुधारली",
        ],
    },
    "Crime": {
        "keywords": [
            "गुन्हा", "पोलिस", "चोरी", "अटक", "खून", "दरोडा", "आरोपी",
            "जेल", "तपास", "फसवणूक", "गुन्हेगार", "हल्ला",
        ],
        "templates": [
            "{place} मध्ये मोठ्या {kw} प्रकरणाचा पोलिसांकडून पर्दाफाश",
            "{kw} प्रकरणी दोन जणांना {kw2}",
            "{place} पोलिसांची {kw} रॅकेटवर कारवाई",
            "बहुचर्चित {kw} प्रकरणाचा तपास अंतिम टप्प्यात",
            "{place} मध्ये {kw} च्या घटनेने खळबळ",
            "सायबर {kw} प्रकरणी सायबर सेलकडून चौकशी",
            "{kw} प्रकरणातील आरोपी पोलिसांच्या ताब्यात",
        ],
    },
    "Tech": {
        "keywords": [
            "तंत्रज्ञान", "इंटरनेट", "मोबाईल", "अॅप", "संगणक", "आयफोन",
            "डेटा", "एआय", "फोन", "शेअर", "बाजार", "गुंतवणूक", "सेन्सेक्स",
            "निफ्टी", "सॉफ्टवेअर", "स्टार्टअप", "कंपनी",
        ],
        "templates": [
            "नवीन {kw} बाजारात दाखल",
            "{kw} क्षेत्रातील मोठ्या {kw2} ची घोषणा",
            "{place} मधील स्टार्टअपने आणले नवे {kw}",
            "{kw} तंत्रज्ञानामुळे उद्योगात मोठे बदल",
            "आजच्या व्यवहारात {kw} मध्ये मोठी वाढ",
            "{kw} लाँच करत कंपनीने वाढवली स्पर्धा",
            "{place} मध्ये {kw} विषयक परिषदेचे आयोजन",
        ],
    },
    "Travel": {
        "keywords": [
            "पर्यटन", "प्रवास", "विमान", "ट्रेन", "बस", "सुट्टी", "ठिकाण",
            "टूर", "समुद्रकिनारा", "हॉटेल", "प्रवासी", "गड",
        ],
        "templates": [
            "{place} हे {kw} साठी सर्वोत्तम ठिकाण",
            "सुट्टीत {place} येथे {kw} साठी गर्दी",
            "नवीन {kw} मार्गामुळे प्रवाशांना फायदा",
            "{place} च्या {kw} स्थळाला पर्यटकांची पसंती",
            "{kw} विभागाकडून नवीन योजना जाहीर",
            "{place} ते {kw} प्रवास आता होणार सोपा",
            "हिवाळी {kw} साठी {place} सज्ज",
        ],
    },
    "Education": {
        "keywords": [
            "शिक्षण", "शाळा", "कॉलेज", "परीक्षा", "निकाल", "विद्यार्थी",
            "अभ्यासक्रम", "विद्यापीठ", "शिष्यवृत्ती", "प्रवेश", "शिक्षक",
        ],
        "templates": [
            "{place} विद्यापीठाचा {kw} जाहीर",
            "{kw} प्रक्रियेला {place} मध्ये सुरुवात",
            "नवीन {kw} धोरणामुळे विद्यार्थ्यांना फायदा",
            "{place} च्या शाळांमध्ये {kw} विषयी विशेष उपक्रम",
            "{kw} परीक्षेचे वेळापत्रक जाहीर",
            "गुणवंत विद्यार्थ्यांना {kw} जाहीर",
            "{place} मध्ये {kw} विषयावर कार्यशाळा",
        ],
    },
    "Fashion": {
        "keywords": [
            "फॅशन", "ड्रेस", "स्टाईल", "सौंदर्य", "कपडे", "सौंदर्यप्रसाधने",
            "ट्रेंड", "मॉडेल", "डिझायनर",
        ],
        "templates": [
            "यंदाच्या हंगामातील नवा {kw} ट्रेंड",
            "{place} मध्ये {kw} शोचे आयोजन",
            "प्रसिद्ध {kw} यांचा नवा कलेक्शन सादर",
            "हिवाळ्यातील {kw} टिप्स तज्ज्ञांकडून",
            "{kw} क्षेत्रात नवीन डिझाईन्सची चर्चा",
            "{place} फॅशन वीकमध्ये {kw} ची धूम",
        ],
    },
    "Bhakti": {
        "keywords": [
            "मंदिर", "देव", "पूजा", "उत्सव", "भजन", "तीर्थक्षेत्र",
            "आरती", "यात्रा", "दर्शन", "कीर्तन", "गणपती", "नवरात्र",
        ],
        "templates": [
            "{place} च्या {kw} मध्ये भाविकांची मोठी गर्दी",
            "यंदाच्या {kw} उत्सवाची जय्यत तयारी",
            "{place} येथे {kw} सोहळा उत्साहात संपन्न",
            "{kw} निमित्त विशेष कार्यक्रमाचे आयोजन",
            "{place} तीर्थक्षेत्री भाविकांसाठी {kw} ची व्यवस्था",
            "पारंपरिक {kw} सोहळ्याला सुरुवात",
        ],
    },
    "Auto": {
        "keywords": [
            "कार", "बाईक", "वाहन", "गाडी", "इंजिन", "मायलेज", "ऑटो",
            "ड्रायव्हिंग", "इलेक्ट्रिक", "स्कूटर", "टायर",
        ],
        "templates": [
            "नवीन {kw} बाजारात लाँच",
            "{kw} च्या किमतीत मोठी घट",
            "{place} मध्ये {kw} प्रदर्शनाचे आयोजन",
            "{kw} खरेदीवर यंदा विशेष सवलत",
            "इलेक्ट्रिक {kw} ला ग्राहकांची पसंती",
            "{kw} च्या नव्या मॉडेलचे अनावरण",
            "{place} मध्ये {kw} सुरक्षा नियम कडक",
        ],
    },
    "International": {
        "keywords": [
            "आंतरराष्ट्रीय", "अमेरिका", "चीन", "युद्ध", "परदेश", "रशिया",
            "जगात", "हवामान", "पाऊस", "थंडी", "उष्णता", "संयुक्त राष्ट्र",
        ],
        "templates": [
            "{kw} स्तरावर मोठा निर्णय जाहीर",
            "{kw} मध्ये तणाव वाढल्याने चिंता",
            "जगभरात {kw} बाबत चर्चा",
            "{kw} परिषदेत महत्त्वाचे मुद्दे मांडले",
            "{kw} बदलामुळे {place} वरही परिणाम",
            "संयुक्त राष्ट्रांकडून {kw} बाबत निवेदन",
        ],
    },
}


def build_rows(per_category=70):
    rows = []
    for category, spec in CATEGORY_DATA.items():
        keywords = spec["keywords"]
        templates = spec["templates"]
        seen = set()
        attempts = 0
        while len(seen) < per_category and attempts < per_category * 20:
            attempts += 1
            template = random.choice(templates)
            kw = random.choice(keywords)
            kw2 = random.choice(keywords)
            place = random.choice(PLACES)
            headline = template.format(kw=kw, kw2=kw2, place=place)
            if headline not in seen:
                seen.add(headline)
                rows.append((headline, category))
    random.shuffle(rows)
    return rows


def main():
    rows = build_rows(per_category=70)
    out_path = "data/marathi_news_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["headline", "category"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    counts = {}
    for _, c in rows:
        counts[c] = counts.get(c, 0) + 1
    for c, n in sorted(counts.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
