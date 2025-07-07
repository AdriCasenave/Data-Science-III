#!/usr/bin/env python3
"""
Script principal para ejecutar el proyecto completo de NLP
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar módulos locales
from nlp_utils import (
    TextPreprocessor, 
    TextAnalyzer, 
    create_sample_dataset, 
    download_nltk_resources,
    quick_text_analysis
)
from model_training import (
    TraditionalMLTrainer,
    DeepLearningTrainer,
    train_complete_pipeline,
    TENSORFLOW_AVAILABLE
)

def setup_environment():
    """Configura el entorno de trabajo"""
    print("🔧 Configurando entorno...")
    
    # Crear directorios necesarios
    dirs_to_create = ['data', 'data/raw', 'data/processed', 'data/models', 'results']
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)
    
    # Descargar recursos de NLTK
    download_nltk_resources()
    
    print("✅ Entorno configurado correctamente")

def load_or_create_dataset():
    """Carga o crea un dataset para el análisis"""
    print("📊 Cargando dataset...")
    
    # Intentar cargar dataset real
    dataset_paths = [
        'data/raw/sentiment140.csv',
        'data/raw/tweets.csv',
        'data/raw/reviews.csv'
    ]
    
    df = None
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"✅ Dataset cargado desde {path}")
                break
            except Exception as e:
                print(f"⚠️  Error cargando {path}: {e}")
    
    # Si no hay dataset real, crear uno sintético
    if df is None:
        print("💡 Creando dataset sintético...")
        df = create_sample_dataset(1500)  # Dataset más grande
        df.to_csv('data/raw/synthetic_dataset.csv', index=False)
        print("✅ Dataset sintético creado y guardado")
    
    return df

def exploratory_analysis(df):
    """Realiza análisis exploratorio"""
    print("\n🔍 Análisis Exploratorio")
    print("=" * 50)
    
    # Información básica
    print(f"📊 Total de textos: {len(df)}")
    print(f"📏 Longitud promedio: {df['text'].str.len().mean():.1f} caracteres")
    print(f"📝 Palabras promedio: {df['text'].str.split().str.len().mean():.1f}")
    
    # Distribución de sentimientos
    print("\n📈 Distribución de sentimientos:")
    sentiment_counts = df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    # Análisis rápido
    print("\n🔬 Análisis de frecuencia de palabras...")
    word_freq = quick_text_analysis(df['text'].tolist())
    
    return word_freq

def preprocessing_pipeline(df):
    """Ejecuta el pipeline de preprocessing"""
    print("\n🔄 Pipeline de Preprocessing")
    print("=" * 50)
    
    # Crear preprocessor
    preprocessor = TextPreprocessor()
    
    # Aplicar preprocessing
    print("🧹 Aplicando limpieza y normalización...")
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Análisis de texto
    print("📊 Creando análisis de texto...")
    analyzer = TextAnalyzer()
    
    # Análisis de sentimientos con VADER
    print("🎭 Aplicando análisis de sentimientos VADER...")
    df_analyzed = analyzer.analyze_sentiment_dataframe(df_processed)
    
    # Comparar con etiquetas originales
    if 'sentiment_label' in df_analyzed.columns:
        accuracy = (df_analyzed['sentiment_label'] == df_analyzed['vader_sentiment']).mean()
        print(f"📊 Accuracy de VADER vs etiquetas originales: {accuracy:.3f}")
    
    # Guardar datos procesados
    output_path = 'data/processed/dataset_processed.csv'
    df_analyzed.to_csv(output_path, index=False)
    print(f"💾 Dataset procesado guardado en {output_path}")
    
    return df_analyzed

def visualization_analysis(df):
    """Crea visualizaciones del análisis"""
    print("\n📊 Creando Visualizaciones")
    print("=" * 50)
    
    analyzer = TextAnalyzer()
    
    # Distribución de sentimientos
    print("📈 Gráfico de distribución de sentimientos...")
    fig1 = analyzer.plot_sentiment_distribution(df)
    fig1.savefig('results/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Word frequency por sentimiento
    print("🔤 Análisis de frecuencia de palabras...")
    sentiments = df['sentiment_label'].unique()
    
    for sentiment in sentiments:
        if sentiment in df['sentiment_label'].values:
            sentiment_texts = df[df['sentiment_label'] == sentiment]['text_clean'].tolist()
            word_freq = analyzer.get_word_frequency(sentiment_texts, n=15)
            
            # Gráfico de frecuencia
            fig2 = analyzer.plot_word_frequency(word_freq, f"Palabras más frecuentes - {sentiment}")
            fig2.savefig(f'results/word_frequency_{sentiment.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
    
    # Word clouds
    print("☁️  Creando word clouds...")
    for sentiment in sentiments:
        if sentiment in df['sentiment_label'].values:
            sentiment_texts = df[df['sentiment_label'] == sentiment]['text_clean'].tolist()
            text_combined = ' '.join(sentiment_texts)
            
            fig3 = analyzer.create_wordcloud(text_combined, f"Word Cloud - {sentiment}")
            fig3.savefig(f'results/wordcloud_{sentiment.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
    
    print("✅ Visualizaciones guardadas en la carpeta 'results'")

def machine_learning_pipeline(df):
    """Ejecuta el pipeline de machine learning"""
    print("\n🤖 Pipeline de Machine Learning")
    print("=" * 50)
    
    # Preparar datos
    texts = df['text'].tolist()
    labels = df['sentiment_label'].tolist()
    
    # Ejecutar pipeline completo
    results = train_complete_pipeline(texts, labels, save_path='data/models')
    
    # Mostrar resultados
    print("\n📊 Resultados Finales:")
    print("-" * 30)
    best_model = None
    best_accuracy = 0
    
    for model_name, result in results.items():
        accuracy = result['accuracy']
        print(f"{model_name:25}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print(f"\n🏆 Mejor modelo: {best_model} ({best_accuracy:.4f})")
    
    return results, best_model

def generate_report(df, results, best_model, word_freq):
    """Genera un reporte final"""
    print("\n📝 Generando Reporte Final")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Reporte de Análisis de Sentimientos
Generado el: {timestamp}

## Resumen del Dataset
- Total de textos: {len(df):,}
- Longitud promedio: {df['text'].str.len().mean():.1f} caracteres
- Palabras promedio: {df['text'].str.split().str.len().mean():.1f}

## Distribución de Sentimientos
"""
    
    sentiment_counts = df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        report += f"- {sentiment}: {count} ({percentage:.1f}%)\n"
    
    report += f"""
## Palabras Más Frecuentes
"""
    
    for i, (word, count) in enumerate(word_freq[:10], 1):
        report += f"{i}. {word}: {count}\n"
    
    report += f"""
## Resultados de Modelos
"""
    
    for model_name, result in results.items():
        report += f"- {model_name}: {result['accuracy']:.4f}\n"
    
    report += f"""
## Mejor Modelo
🏆 {best_model} con accuracy de {results[best_model]['accuracy']:.4f}

## Archivos Generados
- Dataset procesado: data/processed/dataset_processed.csv
- Modelos entrenados: data/models/
- Visualizaciones: results/

## Próximos Pasos
1. Optimizar hiperparámetros del mejor modelo
2. Probar con embeddings pre-entrenados
3. Implementar validación cruzada
4. Crear interfaz web para predicciones
5. Evaluar en datos de producción
"""
    
    # Guardar reporte
    with open('results/reporte_final.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Reporte guardado en 'results/reporte_final.md'")
    
    return report

def main():
    """Función principal"""
    print("🚀 PROYECTO NLP - ANÁLISIS DE SENTIMIENTOS")
    print("=" * 60)
    print("Implementación completa: NLP Tradicional + Deep Learning")
    print("=" * 60)
    
    try:
        # 1. Configurar entorno
        setup_environment()
        
        # 2. Cargar dataset
        df = load_or_create_dataset()
        
        # 3. Análisis exploratorio
        word_freq = exploratory_analysis(df)
        
        # 4. Preprocessing
        df_processed = preprocessing_pipeline(df)
        
        # 5. Visualizaciones
        visualization_analysis(df_processed)
        
        # 6. Machine Learning
        results, best_model = machine_learning_pipeline(df_processed)
        
        # 7. Reporte final
        report = generate_report(df_processed, results, best_model, word_freq)
        
        print("\n🎉 PROYECTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print("📁 Archivos generados:")
        print("  - data/processed/dataset_processed.csv")
        print("  - data/models/ (modelos entrenados)")
        print("  - results/ (visualizaciones y reporte)")
        print("  - results/reporte_final.md")
        
        print(f"\n🏆 Mejor modelo: {best_model}")
        print(f"📊 Accuracy: {results[best_model]['accuracy']:.4f}")
        
        if TENSORFLOW_AVAILABLE:
            print("🧠 Deep Learning: Disponible")
        else:
            print("⚠️  Deep Learning: No disponible (instalar TensorFlow)")
        
        print("\n💡 Para ejecutar los notebooks interactivos:")
        print("   jupyter notebook notebooks/")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
