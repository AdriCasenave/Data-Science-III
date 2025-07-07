"""
Módulo para entrenamiento y evaluación de modelos de Machine Learning y Deep Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# TensorFlow para Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow no disponible. Solo se usarán modelos tradicionales.")

class TraditionalMLTrainer:
    """Clase para entrenar modelos tradicionales de ML"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """Prepara los datos para entrenamiento"""
        # Codificar labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, 
            random_state=random_state, stratify=encoded_labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name, X_train, y_train, **kwargs):
        """Entrena un modelo específico"""
        # Configurar pipelines
        if model_name == 'naive_bayes':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', MultinomialNB())
            ])
        elif model_name == 'logistic_regression':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
        elif model_name == 'svm':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', SVC(random_state=42, probability=True))
            ])
        elif model_name == 'random_forest':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
                ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
            ])
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        # Entrenar
        print(f"🔄 Entrenando {model_name}...")
        pipeline.fit(X_train, y_train)
        
        # Guardar modelo
        self.models[model_name] = pipeline
        print(f"✅ {model_name} entrenado exitosamente")
        
        return pipeline
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evalúa un modelo"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def train_all_models(self, X_train, y_train):
        """Entrena todos los modelos disponibles"""
        models_to_train = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
        
        for model_name in models_to_train:
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                print(f"❌ Error entrenando {model_name}: {e}")
    
    def compare_models(self, X_test, y_test):
        """Compara todos los modelos entrenados"""
        results = {}
        
        for model_name in self.models:
            try:
                result = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = result
                print(f"{model_name}: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"❌ Error evaluando {model_name}: {e}")
        
        return results
    
    def save_models(self, path='../data/models'):
        """Guarda todos los modelos entrenados"""
        os.makedirs(path, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(path, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"✅ {model_name} guardado en {model_path}")
        
        # Guardar label encoder
        if self.label_encoder:
            encoder_path = os.path.join(path, "label_encoder.pkl")
            joblib.dump(self.label_encoder, encoder_path)

class DeepLearningTrainer:
    """Clase para entrenar modelos de Deep Learning"""
    
    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está disponible")
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.history = None
        self.config = {}
        
    def prepare_data(self, texts, labels, max_features=5000, max_length=100, test_size=0.2, random_state=42):
        """Prepara datos para Deep Learning"""
        # Codificar labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        categorical_labels = to_categorical(encoded_labels, num_classes=num_classes)
        
        # Tokenizar textos
        self.tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, categorical_labels, test_size=test_size, 
            random_state=random_state, stratify=encoded_labels
        )
        
        # Crear conjunto de validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state,
            stratify=np.argmax(y_train, axis=1)
        )
        
        # Guardar configuración
        self.config = {
            'max_features': max_features,
            'max_length': max_length,
            'num_classes': num_classes,
            'vocab_size': min(max_features, len(self.tokenizer.word_index) + 1)
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_lstm_model(self, embedding_dim=128, lstm_units=64, dropout_rate=0.3):
        """Crea un modelo LSTM"""
        model = Sequential([
            Embedding(self.config['vocab_size'], embedding_dim, input_length=self.config['max_length']),
            Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)),
            Bidirectional(LSTM(lstm_units // 2, dropout=dropout_rate)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.config['num_classes'], activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, model_type='lstm', epochs=20, batch_size=32):
        """Entrena el modelo"""
        # Crear modelo
        if model_type == 'lstm':
            self.model = self.create_lstm_model()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        print(f"🔄 Entrenando modelo {model_type}...")
        print(f"Parámetros: {self.model.count_params():,}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
        ]
        
        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evalúa el modelo"""
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        # Evaluar
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Métricas
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, path='../data/models'):
        """Guarda el modelo y componentes"""
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelo
        model_path = os.path.join(path, 'deep_learning_model.h5')
        self.model.save(model_path)
        
        # Guardar tokenizer
        import pickle
        tokenizer_path = os.path.join(path, 'tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Guardar label encoder
        encoder_path = os.path.join(path, 'dl_label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ Modelo guardado en {path}")

def train_complete_pipeline(texts, labels, save_path='../data/models'):
    """Entrena una pipeline completa con modelos tradicionales y deep learning"""
    print("🚀 Iniciando pipeline completo de entrenamiento")
    
    results = {}
    
    # 1. Modelos tradicionales
    print("\n=== MODELOS TRADICIONALES ===")
    ml_trainer = TraditionalMLTrainer()
    X_train, X_test, y_train, y_test = ml_trainer.prepare_data(texts, labels)
    
    # Entrenar todos los modelos
    ml_trainer.train_all_models(X_train, y_train)
    
    # Evaluar modelos
    ml_results = ml_trainer.compare_models(X_test, y_test)
    results.update(ml_results)
    
    # Guardar modelos tradicionales
    ml_trainer.save_models(save_path)
    
    # 2. Deep Learning (si está disponible)
    if TENSORFLOW_AVAILABLE:
        print("\n=== DEEP LEARNING ===")
        try:
            dl_trainer = DeepLearningTrainer()
            
            # Preparar datos
            X_train_dl, X_val_dl, X_test_dl, y_train_dl, y_val_dl, y_test_dl = dl_trainer.prepare_data(texts, labels)
            
            # Entrenar modelo LSTM
            dl_trainer.train_model(X_train_dl, y_train_dl, X_val_dl, y_val_dl, model_type='lstm', epochs=10)
            
            # Evaluar
            dl_results = dl_trainer.evaluate_model(X_test_dl, y_test_dl)
            results['deep_learning_lstm'] = {
                'accuracy': dl_results['test_accuracy'],
                'predictions': dl_results['predictions'],
                'probabilities': dl_results['probabilities'],
                'classification_report': dl_results['classification_report'],
                'confusion_matrix': dl_results['confusion_matrix']
            }
            
            # Guardar modelo
            dl_trainer.save_model(save_path)
            
        except Exception as e:
            print(f"❌ Error en Deep Learning: {e}")
    
    # 3. Comparación final
    print("\n=== RESUMEN DE RESULTADOS ===")
    for model_name, result in results.items():
        print(f"{model_name:20}: {result['accuracy']:.4f}")
    
    return results
