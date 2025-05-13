import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from emotion_detection import create_cbam_4cnn, CBAMBlock
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import cv2

def load_data(data_dir='fer2013'):
    """Cargar los datos de entrenamiento"""
    X = []
    y = []
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    
    print("Cargando datos...")
    
    # Verificar si existe el directorio fer2013
    if not os.path.exists(data_dir):
        print(f"Error: El directorio {data_dir} no existe.")
        return None, None, None, None
    
    # Verificar si existe la carpeta train
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: No se encontró la carpeta 'train' en {data_dir}")
        return None, None, None, None
    
    # Contar imágenes por emoción
    emotion_counts = {emotion: 0 for emotion in emotions}
    
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(train_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Advertencia: Directorio {emotion_dir} no encontrado")
            continue
            
        print(f"Procesando emoción: {emotion}")
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (48, 48))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)
                
                X.append(img)
                y.append(emotion_idx)
                emotion_counts[emotion] += 1
            except Exception as e:
                print(f"Error procesando {img_path}: {str(e)}")
                continue
    
    # Verificar si se cargaron imágenes
    if len(X) == 0:
        print("Error: No se encontraron imágenes en el directorio fer2013/train.")
        return None, None, None, None
    
    # Mostrar estadísticas de imágenes cargadas
    print("\nEstadísticas de imágenes cargadas:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} imágenes")
    
    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    # Convertir y a one-hot encoding
    y = np.eye(len(emotions))[y]
    
    # Dividir en train y validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTotal de imágenes cargadas: {len(X)}")
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
    print(f"Tamaño del conjunto de validación: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val

def train_model():
    """Entrenar el modelo por 5 épocas"""
    # Cargar datos
    X_train, y_train, X_val, y_val = load_data()
    if X_train is None:
        return
    
    # Crear el modelo con la arquitectura exacta
    model = create_cbam_4cnn(input_shape=(48, 48, 1), num_classes=8)
    
    # Configurar callback para guardar el mejor modelo
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Entrenar el modelo por 5 épocas
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=[checkpoint]
    )
    
    # Imprimir estadísticas del entrenamiento
    print("\nEstadísticas del entrenamiento:")
    print(f"Precisión final de entrenamiento: {history.history['accuracy'][-1]:.4f}")
    print(f"Precisión final de validación: {history.history['val_accuracy'][-1]:.4f}")
    
    # Guardar el modelo final
    model.save('best_model.h5')
    print("Modelo guardado como 'best_model.h5'")

if __name__ == "__main__":
    # Configurar GPU si está disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configurada")
    
    train_model() 