import re
import string
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Pre-initialize the normalizer for efficiency
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("mr")

# ---------------------------------------------------------------------------
# EXTENDED MARATHI STOPWORDS  (250+ entries)
# ---------------------------------------------------------------------------
MARATHI_STOPWORDS = {
    # Conjunctions / connectives
    "आणि", "अथवा", "किंवा", "की", "तर", "तरी", "परंतु", "पण", "व",
    "तसेच", "म्हणजे", "म्हणून", "कारण", "जेव्हा", "जेथे", "जसे",
    "जरी", "जिथे", "जिथून", "तेव्हा", "तेथे", "तसे", "जणू",
    # Pronouns
    "मी", "तू", "तो", "ती", "ते", "आम्ही", "आपण", "तुम्ही",
    "त्यांनी", "त्याला", "त्याचा", "त्याची", "त्याचे", "त्यांना",
    "त्यांचा", "त्यांची", "त्यांचे", "तिला", "तिचा", "तिची", "तिचे",
    "यांनी", "यांना", "यांचा", "यांची", "यांचे",
    "ज्यांना", "ज्यांचा", "ज्यांची", "ज्यांचे",
    "स्वतः", "आपला", "आपली", "आपले",
    # Demonstratives / articles
    "हा", "ही", "हे", "या", "त्या", "एक", "एका", "काही", "कोण",
    "कोणी", "कोणता", "कोणती", "कोणते", "कुणी", "कुठे", "कुठून",
    "कुठला", "काय", "केव्हा", "कसे", "कशा", "कशाने",
    # Postpositions / case-markers
    "ला", "ना", "ने", "नी", "त", "मध्ये", "मधे", "वर", "वरून",
    "खाली", "साठी", "बद्दल", "विषयी", "विरुद्ध", "संबंधी", "पेक्षा",
    "पासून", "पर्यंत", "आधी", "नंतर", "मुळे", "कडे", "कडून", "हून",
    "मधून", "प्रमाणे", "सारखे", "सोबत", "बरोबर", "शिवाय", "अगोदर",
    "कडील", "मधील", "वरील",
    # Adverbs & particles
    "अगदी", "आता", "अजून", "आजही", "केवळ", "फक्त", "मात्र",
    "खरोखर", "खूप", "फार", "इतके", "जास्त", "कमी", "सर्वात",
    "नक्की", "नेहमी", "नेमके", "नुकतेच", "इथे", "तिथे", "सर्वत्र",
    "अनेक", "बरेच", "थोडे", "सध्या", "लवकर", "उशिरा", "पुन्हा",
    "पुढे", "मागे", "बाहेर", "आत", "जवळ", "दूर",
    # Auxiliaries / light verbs
    "आहे", "आहेत", "होता", "होती", "होते", "होतो", "होतात",
    "असतो", "असते", "असतात", "असेल", "असतील", "नाही", "नाहीत",
    "नव्हते", "नव्हती", "नव्हता", "केला", "केली", "केले",
    "करतो", "करते", "करतात", "केलेला", "केलेली", "केलेले",
    "करायला", "करणे", "करत", "करून",
    "घेणे", "घेतले", "घेतला", "घेतली", "दिले", "दिला", "दिली",
    "सांगणे", "सांगितले", "येणे", "आले", "आला", "आली",
    "जाणे", "गेला", "गेली", "गेले", "पाहणे", "पाहिले",
    "मिळणे", "मिळाले", "झाले", "झाला", "झाली", "होणे",
    # Common filler nouns
    "लोक", "जण", "वेळ", "वेळी", "दिवस", "रात्र", "काम",
    "गोष्ट", "बाब", "प्रकार", "जागी", "बाजू", "दरम्यान",
    # Quantifiers
    "दोन", "तीन", "सर्व", "बरेच", "थोडे",
    # Miscellaneous functional words
    "इतर", "इन", "उर्फ", "वगैरे", "वा", "च", "का", "हो",
    "अर्थात", "अशा", "अशाप्रकारे", "असे", "येथे",
    # Postfix particles
    "च्या", "चे", "ची", "चा", "सुद्धा", "देखील",
    # Negation & modal
    "नये", "नसून", "नसे", "नसता", "नसताना",
    # Interrogatives
    "कधी", "किती",
}

# ---------------------------------------------------------------------------
# EXTENDED MARATHI SUFFIXES  (longest-first for greedy matching)
# ---------------------------------------------------------------------------
MARATHI_SUFFIXES = [
    # 8+ chars
    "करण्यासाठी",
    # 7 chars
    "करण्यास", "प्रमाणेच",
    # 6 chars
    "प्रमाणे", "पर्यंत", "बद्दलचे", "विषयीचे",
    # 5 chars
    "साठी", "मधून", "कडून", "कडील", "मधील", "वरून", "वरील", "पासून",
    "णारा", "णारी", "णारे", "लेला", "लेली", "लेले", "ताना",
    # 4 chars
    "च्या", "मधे", "कडे", "मुळे", "हून", "पेक्षा", "सुद्धा", "पैकी",
    "णे", "वून",
    # 3 chars
    "चा", "ची", "चे", "ने", "नी", "ही", "ला", "ते", "ना",
    "तून", "तात", "तो",
    # 2 chars
    "त", "स", "शी", "ऊन",
]

# Deduplicate, sort longest-first
_seen_suf: set = set()
_ordered: list = []
for _s in MARATHI_SUFFIXES:
    if _s not in _seen_suf:
        _seen_suf.add(_s)
        _ordered.append(_s)
MARATHI_SUFFIXES = sorted(_ordered, key=len, reverse=True)


def simple_marathi_stemmer(word: str) -> str:
    """Strip the longest matching suffix, keeping a meaningful stem."""
    for suffix in MARATHI_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


# ---------------------------------------------------------------------------
# STEP-BY-STEP PREPROCESSOR
# ---------------------------------------------------------------------------

def preprocess_steps(text: str) -> dict:
    """
    Run each preprocessing stage and return a dict with both the intermediate
    outputs (for display) and the final cleaned string (for the model).

    Stages
    ------
    1. Raw input
    2. Normalization  - IndicNLP normalizer (ZWJ, ZWNJ, Nukta, etc.)
    3. Tokenization   - Indic trivial tokenizer
    4. Punctuation removal
    5. Stopword removal
    6. Stemming / suffix stripping
    """
    if not text:
        return {
            "step1_raw": "",
            "step2_normalized": "",
            "step3_tokens": [],
            "step4_no_punct": [],
            "step5_no_stopwords": [],
            "step6_stemmed": [],
            "final": "",
        }

    # 1. Raw
    raw = text.strip()

    # 2. Normalization
    normalized = normalizer.normalize(raw)

    # 3. Tokenization
    tokens = indic_tokenize.trivial_tokenize(normalized)

    # 4. Punctuation removal (also strips pure-digit tokens & lone symbols)
    punct_set = set(string.punctuation + "।॥॰")
    no_punct = [
        w for w in tokens
        if w not in punct_set and not re.fullmatch(r"[\d\s]+", w)
    ]

    # 5. Stopword removal
    no_stopwords = [w for w in no_punct if w not in MARATHI_STOPWORDS]

    # 6. Stemming
    stemmed = [simple_marathi_stemmer(w) for w in no_stopwords]

    return {
        "step1_raw": raw,
        "step2_normalized": normalized,
        "step3_tokens": tokens,
        "step4_no_punct": no_punct,
        "step5_no_stopwords": no_stopwords,
        "step6_stemmed": stemmed,
        "final": " ".join(stemmed),
    }


def preprocess(text: str) -> str:
    """
    Backward-compatible entry-point used by train_model.py and app.py.
    Returns the final space-separated stemmed string.
    """
    return preprocess_steps(text)["final"]
