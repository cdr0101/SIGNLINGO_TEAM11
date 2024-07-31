from googletrans import Translator  # Import the Translator from googletrans
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles import finders
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import pipeline

nltk.download('stopwords')

def home_view(request):
    return render(request, 'home.html')

def knowUs_view(request):
    return render(request, 'knowUs.html')

emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_emoji_map = {
    'joy': '😊',
    'love': '❤️',
    'sad': '😢',
    'anger': '😠',
    'surprise': '😮',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐',
    'happy': '😊',
    }

def text2sign_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        print("Input Text:", text)  # Debug: Print the input text
        
        # Translate text to English using googletrans
        translator = Translator()
        translation = translator.translate(text, dest='en')
        translated_text = translation.text
        print("Translated Text:", translated_text)  # Debug: Print the translated text

        # Tokenizing the translated text
        translated_text_lower = translated_text.lower()
        words = word_tokenize(translated_text_lower)
        tagged = nltk.pos_tag(words)
        tense = {
            "future": len([word for word in tagged if word[1] == "MD"]),
            "present": len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]]),
            "past": len([word for word in tagged if word[1] in ["VBD", "VBN"]]),
            "present_continuous": len([word for word in tagged if word[1] in ["VBG"]])
        }

        stop_words = set(stopwords.words('english'))
        lr = WordNetLemmatizer()
        filtered_text = [
            lr.lemmatize(w, pos='v') if p[1].startswith('V') else lr.lemmatize(w)
            for w, p in zip(words, tagged) if w not in stop_words
        ]

        words = ['Me' if w == 'I' else w for w in filtered_text]
        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            words.insert(0, "Before")
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in words:
                words.insert(0, "Will")
        elif probable_tense == "present" and tense["present_continuous"] >= 1:
            words.insert(0, "Now")

        filtered_text = []
        for w in words:
            path = w + ".mp4"
            if not finders.find(path):
                filtered_text.extend(list(w))
            else:
                filtered_text.append(w)
        words = filtered_text

        # Emotion recognition
        emotion = emotion_model(translated_text_lower)[0]
        emotion_label = emotion['label']
        # emotion_score = emotion['score']
        emotion_emoji = emotion_emoji_map.get(emotion_label.lower(), '😐')  # Default to neutral emoji if not found

        return render(request, 'text2sign.html', {
            'words': words,
            'text': translated_text,  # Use translated text for display
            'emotion_label': emotion_label,
            'emotion_emoji': emotion_emoji,
            # 'emotion_score': emotion_score
        })
    else:
        return render(request, 'text2sign.html')

# # Initialize emotion recognition pipeline
# emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# def text2sign_view(request):
#     if request.method == 'POST':
#         text = request.POST.get('sen')
#         print("Input Text:", text)  # Debug: Print the input text
        
#         # Translate text to English using googletrans
#         translator = Translator()
#         translation = translator.translate(text, dest='en')
#         translated_text = translation.text
#         print("Translated Text:", translated_text)  # Debug: Print the translated text

#         # Tokenizing the translated text
#         translated_text_lower = translated_text.lower()
#         words = word_tokenize(translated_text_lower)
#         tagged = nltk.pos_tag(words)
#         tense = {
#             "future": len([word for word in tagged if word[1] == "MD"]),
#             "present": len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]]),
#             "past": len([word for word in tagged if word[1] in ["VBD", "VBN"]]),
#             "present_continuous": len([word for word in tagged if word[1] in ["VBG"]])
#         }

#         stop_words = set(stopwords.words('english'))
#         lr = WordNetLemmatizer()
#         filtered_text = [
#             lr.lemmatize(w, pos='v') if p[1].startswith('V') else lr.lemmatize(w)
#             for w, p in zip(words, tagged) if w not in stop_words
#         ]

#         words = ['Me' if w == 'I' else w for w in filtered_text]
#         probable_tense = max(tense, key=tense.get)

#         if probable_tense == "past" and tense["past"] >= 1:
#             words.insert(0, "Before")
#         elif probable_tense == "future" and tense["future"] >= 1:
#             if "Will" not in words:
#                 words.insert(0, "Will")
#         elif probable_tense == "present" and tense["present_continuous"] >= 1:
#             words.insert(0, "Now")

#         filtered_text = []
#         for w in words:
#             path = w + ".mp4"
#             if not finders.find(path):
#                 filtered_text.extend(list(w))
#             else:
#                 filtered_text.append(w)
#         words = filtered_text

#         # Emotion recognition
#         emotion = emotion_model(translated_text_lower)[0]

#         return render(request, 'text2sign.html', {
#             'words': words,
#             'text': translated_text,  # Use translated text for display
#             'emotion_label': emotion['label'],
#             'emotion_score': emotion['score']
#         })
#     else:
#         return render(request, 'text2sign.html')


def sign2text_view(request):
	return render(request,'sign2text.html')
