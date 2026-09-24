import string
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Pre-initialize the normalizer for efficiency
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("mr")

MARATHI_STOPWORDS = {
    "अथवा", "आणि", "आहे", "आहेत", "इतके", "इतर", "इन", "उर्फ", "एक", "का", "काही", 
    "किंवा", "की", "केला", "केली", "केले", "कोणी", "तर", "तरी", "तसेच", "ती", "ते", 
    "तो", "त्या", "त्याचा", "त्याची", "त्याचे", "त्यांना", "त्याला", "पण", "परंतु", 
    "फार", "मग", "मात्र", "मी", "या", "ला", "वर", "वगैरे", "वा", "शिवाजी", "सर्व", 
    "साठी", "होता", "होती", "होते", "हे", "हा", "ही", "च्या", "चे", "ची", "चा", 
    "आजही", "करतात", "नाही", "नये", "नसून", "नसे", "व", "त्यांनी"
}

MARATHI_SUFFIXES = [
    "चा", "ची", "चे", "च्या", "ला", "त", "ने", "नी", "वर", "साठी", "मधून", 
    "कडे", "हून", "मुळे", "प्रमाणे", "पर्यंत", "स", "शी", "पैकी", "सुद्धा", "ही"
]

def simple_marathi_stemmer(word):
    for suffix in MARATHI_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    return word

def preprocess(text: str) -> str:
    """
    Advanced Marathi text preprocessing:
    1. Normalization (fixes zero-width joiners, etc.)
    2. Word Tokenization
    3. Punctuation Removal
    4. Stopword Removal
    5. Rule-based Stemming
    """
    if not text:
        return ""
        
    # 1. Normalization
    normalized_text = normalizer.normalize(text.strip())
    
    # 2. Tokenization
    words = indic_tokenize.trivial_tokenize(normalized_text)
    
    # 3. Punctuation Removal
    punctuations = set(string.punctuation + "।")
    words_no_punct = [word for word in words if word not in punctuations]
    
    # 4. Stopword Removal
    words_no_stopwords = [word for word in words_no_punct if word not in MARATHI_STOPWORDS]
    
    # 5. Stemming
    stemmed_words = [simple_marathi_stemmer(word) for word in words_no_stopwords]
    
    # Return as space-separated string for compatibility with TF-IDF Vectorizer
    return " ".join(stemmed_words)
