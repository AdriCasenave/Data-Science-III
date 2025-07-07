"""
Utilidades para preprocessamiento de texto y análisis de NLP
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Descargar recursos necesarios
def download_nltk_resources():
    """Descarga recursos necesarios de NLTK"""
    resources = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

class TextPreprocessor:
    """Clase para preprocessamiento de texto"""
    
    def __init__(self, language='english'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
    def clean_text(self, text):
        """Limpia y normaliza el texto"""
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokeniza y aplica lemmatización"""
        # Tokenizar
        tokens = word_tokenize(text)
        
        # Eliminar stopwords y palabras muy cortas
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatización
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text):
        """Aplica todo el preprocessing"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text'):
        """Preprocessa una columna de texto en un DataFrame"""
        df = df.copy()
        df['text_clean'] = df[text_column].apply(self.preprocess_text)
        return df

class TextAnalyzer:
    """Clase para análisis de texto"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def get_word_frequency(self, texts, n=20):
        """Obtiene las palabras más frecuentes"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        return word_freq.most_common(n)
    
    def get_sentiment_vader(self, text):
        """Obtiene sentiment usando VADER"""
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'Positivo', compound
        elif compound <= -0.05:
            return 'Negativo', compound
        else:
            return 'Neutral', compound
    
    def analyze_sentiment_dataframe(self, df, text_column='text'):
        """Analiza sentiment en un DataFrame"""
        df = df.copy()
        
        # Aplicar análisis de sentimiento
        sentiment_results = df[text_column].apply(self.get_sentiment_vader)
        df['vader_sentiment'] = [result[0] for result in sentiment_results]
        df['vader_score'] = [result[1] for result in sentiment_results]
        
        return df
    
    def create_wordcloud(self, text, title="Word Cloud", colormap='viridis'):
        """Crea un word cloud"""
        if isinstance(text, list):
            text = ' '.join(text)
        
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_word_frequency(self, word_freq, title="Palabras más frecuentes", top_n=20):
        """Grafica las palabras más frecuentes"""
        words = [word for word, count in word_freq[:top_n]]
        counts = [count for word, count in word_freq[:top_n]]
        
        plt.figure(figsize=(12, 6))
        plt.barh(words[::-1], counts[::-1], color='skyblue')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Frecuencia')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_sentiment_distribution(self, df, sentiment_column='sentiment_label'):
        """Grafica la distribución de sentimientos"""
        sentiment_counts = df[sentiment_column].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        ax1.set_title('Distribución de Sentimientos')
        ax1.set_xlabel('Sentimiento')
        ax1.set_ylabel('Cantidad')
        
        # Añadir valores sobre las barras
        for bar, value in zip(bars, sentiment_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sentiment_counts.values)*0.01,
                    str(value), ha='center', fontweight='bold')
        
        # Gráfico de torta
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Distribución de Sentimientos (%)')
        
        plt.tight_layout()
        return fig

def create_sample_dataset(size=1500):
    """
    Crea un dataset REALISTA con distribución desbalanceada 
    como se encuentra en datos reales de reviews
    """
    positive_texts = [
        "I love this movie, it's absolutely fantastic!",
        "Amazing performance by all the actors!",
        "Great film with excellent cinematography",
        "Wonderful story and brilliant directing",
        "Outstanding movie, highly recommended!",
        "Perfect entertainment for the whole family",
        "Incredible acting and beautiful soundtrack",
        "This film exceeded all my expectations",
        "Masterpiece of modern cinema",
        "Absolutely loved every minute of it",
        "Best movie I've seen this year",
        "Brilliant storytelling and amazing visuals",
        "Cannot recommend this enough",
        "Fantastic experience from start to finish",
        "Truly exceptional filmmaking at its finest"
    ]
    
    negative_texts = [
        "This movie is terrible, I hate it completely",
        "Worst film I have ever seen in my life",
        "Boring and predictable storyline",
        "Poor acting and bad direction",
        "Complete waste of time and money",
        "Disappointing and poorly executed",
        "Terrible script and awful performances",
        "Not worth watching at all",
        "Very bad movie with no redeeming qualities",
        "Horrible experience, would not recommend",
        "Absolutely terrible waste of time",
        "Poorly written and badly acted",
        "One of the worst movies ever made"
    ]
    
    neutral_texts = [
        "The movie was okay, nothing special",
        "Average film with some good moments",
        "Not bad but could have been better",
        "Decent movie for a casual watch",
        "It was fine, met my expectations",
        "Moderate entertainment value",
        "Acceptable but forgettable",
        "Standard movie with typical plot",
        "Neither good nor bad, just average",
        "Okay for a one-time watch",
        "Pretty standard, nothing groundbreaking",
        "Watchable but not memorable"
    ]
    
    # DISTRIBUCIÓN REALISTA (como datos reales de reviews)
    # Positivo: ~45% (la gente tiende a escribir más reviews positivas)
    # Negativo: ~35% (gente molesta escribe reviews) 
    # Neutral: ~20% (menos común, gente neutral no suele escribir)
    
    positive_count = int(size * 0.45)  # 675 textos
    negative_count = int(size * 0.35)  # 525 textos
    neutral_count = size - positive_count - negative_count  # 300 textos
    
    print(f"📊 Distribución REALISTA del dataset:")
    print(f"  Positivo: {positive_count} ({positive_count/size*100:.1f}%)")
    print(f"  Negativo: {negative_count} ({negative_count/size*100:.1f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/size*100:.1f}%)")
    
    # Generar textos con VARIACIÓN para hacerlo más realista
    all_texts = []
    all_sentiments = []
    all_labels = []
    
    # Textos positivos
    for i in range(positive_count):
        text = positive_texts[i % len(positive_texts)]
        # Añadir pequeñas variaciones
        if i % 3 == 0:
            text = text.replace("movie", "film")
        elif i % 3 == 1:
            text = text.replace("!", ".")
        all_texts.append(text)
        all_sentiments.append(4)
        all_labels.append('Positivo')
    
    # Textos negativos
    for i in range(negative_count):
        text = negative_texts[i % len(negative_texts)]
        if i % 4 == 0:
            text = text.replace("movie", "film")
        all_texts.append(text)
        all_sentiments.append(0)
        all_labels.append('Negativo')
    
    # Textos neutrales
    for i in range(neutral_count):
        text = neutral_texts[i % len(neutral_texts)]
        if i % 2 == 0:
            text = text.replace("movie", "film")
        all_texts.append(text)
        all_sentiments.append(2)
        all_labels.append('Neutral')
    
    # MEZCLAR aleatoriamente (importante para evitar patrones)
    np.random.seed(42)  # Para reproducibilidad en entrega académica
    indices = np.random.permutation(len(all_texts))
    
    final_texts = [all_texts[i] for i in indices]
    final_sentiments = [all_sentiments[i] for i in indices]
    final_labels = [all_labels[i] for i in indices]
    
    df = pd.DataFrame({
        'text': final_texts,
        'sentiment': final_sentiments,
        'sentiment_label': final_labels
    })
    
    # Verificar distribución final
    print(f"\n✅ Dataset REALISTA creado:")
    distribution = df['sentiment_label'].value_counts().sort_index()
    for sentiment, count in distribution.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    print(f"📊 Total: {len(df)} textos")
    
    return df

# Función de utilidad para análisis rápido
def quick_text_analysis(texts, sentiment_labels=None):
    """Realiza un análisis rápido de texto"""
    print("=== ANÁLISIS RÁPIDO DE TEXTO ===")
    
    if isinstance(texts, str):
        texts = [texts]
    
    # Estadísticas básicas
    print(f"Total de textos: {len(texts)}")
    print(f"Longitud promedio: {np.mean([len(text) for text in texts]):.1f} caracteres")
    print(f"Palabras promedio: {np.mean([len(text.split()) for text in texts]):.1f}")
    
    # Análisis de frecuencia
    analyzer = TextAnalyzer()
    word_freq = analyzer.get_word_frequency(texts, n=10)
    
    print("\n=== TOP 10 PALABRAS ===")
    for word, count in word_freq:
        print(f"{word:15} : {count:4d}")
    
    # Análisis de sentimiento si no se proporcionan labels
    if sentiment_labels is None:
        print("\n=== ANÁLISIS DE SENTIMIENTO (VADER) ===")
        sentiments = [analyzer.get_sentiment_vader(text)[0] for text in texts]
        sentiment_counts = Counter(sentiments)
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment:10} : {count:4d}")
    
    return word_freq
