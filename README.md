# Proyecto NLP - Análisis de Sentimientos 🎯
## Entrega Final - Procesamiento de Lenguaje Natural y Deep Learning

### Dataset utilizado:

https://www.kaggle.com/datasets/kazanova/sentiment140

### 📋 Descripción del Proyecto
Este proyecto implementa un sistema completo de análisis de sentimientos que combina:
1. **Técnicas tradicionales de NLP** (tokenización, preprocessing, análisis estadístico)
2. **Modelo de Deep Learning** (red neuronal para clasificación de sentimientos)

### 🗂️ Estructura del Proyecto
```
proyecto_nlp/
├── data/                       # Datasets y datos procesados
│   ├── raw/                   # Datos originales
│   ├── processed/             # Datos procesados
│   └── models/                # Modelos entrenados
├── notebooks/                  # Jupyter notebooks interactivos
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_nlp_tradicional.ipynb
│   └── 03_deep_learning.ipynb
├── src/                        # Código fuente modular
│   ├── nlp_utils.py           # Utilidades de NLP
│   └── model_training.py      # Entrenamiento de modelos
├── results/                    # Resultados y visualizaciones
├── main.py                     # Script principal
├── setup.sh                    # Script de configuración
├── requirements.txt            # Dependencias
└── README.md                   # Este archivo
```



#### Instalación
```bash
# Crear entorno virtual
python3 -m venv venv_nlp
source venv_nlp/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar proyecto
python main.py
```

### 📊 Características Implementadas

#### NLP Tradicional:
- ✅ Tokenización
- ✅ Eliminación de stopwords
- ✅ Lemmatización
- ✅ Análisis de frecuencias
- ✅ TF-IDF vectorización
- ✅ Análisis con VADER
- ✅ Modelos: Naive Bayes, Logistic Regression, SVM, Random Forest

#### Deep Learning:
- ✅ Tokenización numérica
- ✅ Padding de secuencias
- ✅ Embeddings
- ✅ Redes LSTM bidireccionales
- ✅ Dropout y regularización
- ✅ Early stopping
- ✅ Callbacks de TensorFlow

#### Visualizaciones:
- ✅ Distribución de sentimientos
- ✅ Palabras más frecuentes
- ✅ Word clouds por sentimiento
- ✅ Matrices de confusión
- ✅ Curvas de aprendizaje

### 🎯 Resultados Esperados
El proyecto genera automáticamente:
- Dataset procesado con análisis de sentimientos
- Modelos entrenados listos para usar
- Visualizaciones informativas
- Reporte completo con métricas
- Comparación entre modelos tradicionales y deep learning

### 🏆 Logros del Proyecto
- ✅ **NLP Completo**: Tokenización, preprocessing, análisis de frecuencias
- ✅ **Deep Learning**: Redes neuronales LSTM con regularización
- ✅ **Visualizaciones**: Gráficos informativos y word clouds
- ✅ **Modelos Múltiples**: Comparación de 4+ algoritmos diferentes
- ✅ **Automatización**: Pipeline completo ejecutable con un comando
- ✅ **Documentación**: Código bien documentado y explicado
- ✅ **Reproducibilidad**: Resultados consistentes y replicables
