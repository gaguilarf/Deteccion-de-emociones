import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Layer
from tensorflow.keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import multiply, Reshape, Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import os

# Definición de las emociones (usadas en detect_emotion)
EMOTIONS = ['Enojo', 'Desprecio', 'Disgusto', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']

class CBAMBlock(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.ratio = ratio
        
    def build(self, input_shape):
        channel = input_shape[-1]
        # Capas compartidas para Channel Attention
        self.shared_layer_one = Dense(channel//self.ratio, activation='relu', name=f'{self.name}_shared_dense_1')
        self.shared_layer_two = Dense(channel, name=f'{self.name}_shared_dense_2')
        # Capa para Spatial Attention
        self.conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', name=f'{self.name}_spatial_conv')(None, None, input_shape) # Build conv explicitly
        super(CBAMBlock, self).build(input_shape)
        
    def call(self, x):
        # Channel Attention
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalAveragePooling2D()(x)
        
        channel = x.shape[-1]
        # Reshape para aplicar las capas densas
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        max_pool = Reshape((1, 1, channel))(max_pool)
        
        # Aplicar capas compartidas (MLP)
        avg_out = self.shared_layer_two(self.shared_layer_one(avg_pool))
        max_out = self.shared_layer_two(self.shared_layer_one(max_pool))
        
        # Sumar y aplicar sigmoide para obtener pesos de canal
        channel_attention = Activation('sigmoid')(avg_out + max_out)
        
        # Spatial Attention
        # Promedio y máximo a lo largo del eje de canales
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        
        # Concatenar y aplicar convolución 7x7
        concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.conv(concat) # Aplicar la conv 7x7
        
        # Aplicar las atenciones multiplicando
        x = multiply([x, channel_attention])
        x = multiply([x, spatial_attention])
        
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({'ratio': self.ratio})
        return config
    
    # Añadir de_serialize para compatibilidad con load_model si es necesario en el futuro
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_cbam_4cnn(input_shape=(48, 48, 1), num_classes=8):
    """
    Creación del modelo CNN (arquitectura modificada para tener ~10 capas
    serializables, eliminando los bloques CBAM originales para resolver
    el error de Layer count mismatch 10 vs 14).
    """
    inputs = Input(shape=input_shape, name='input_layer')
    
    # Bloque 1 (Conv -> BN -> Act -> Pool -> Drop)
    x = Conv2D(64, (3, 3), padding='same', name='conv2d_1')(inputs)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = Activation('relu', name='activation_1')(x)
    # x = CBAMBlock(name='cbam_block_1')(x) # <--- ELIMINADO
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(x)
    x = Dropout(0.25, name='dropout_1')(x)
    
    # Bloque 2 (Conv -> BN -> Act -> Pool -> Drop)
    x = Conv2D(128, (3, 3), padding='same', name='conv2d_2')(x)
    x = BatchNormalization(name='batch_normalization_2')(x)
    x = Activation('relu', name='activation_2')(x)
    # x = CBAMBlock(name='cbam_block_2')(x) # <--- ELIMINADO
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(x)
    x = Dropout(0.25, name='dropout_2')(x)
    
    # Bloque 3 (Conv -> BN -> Act -> Pool -> Drop)
    x = Conv2D(256, (3, 3), padding='same', name='conv2d_3')(x)
    x = BatchNormalization(name='batch_normalization_3')(x)
    x = Activation('relu', name='activation_3')(x)
    # x = CBAMBlock(name='cbam_block_3')(x) # <--- ELIMINADO
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(x)
    x = Dropout(0.25, name='dropout_3')(x)
    
    # Bloque 4 (Conv -> BN -> Act -> Pool -> Drop)
    x = Conv2D(512, (3, 3), padding='same', name='conv2d_4')(x)
    x = BatchNormalization(name='batch_normalization_4')(x)
    x = Activation('relu', name='activation_4')(x)
    # x = CBAMBlock(name='cbam_block_4')(x) # <--- ELIMINADO
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4')(x)
    x = Dropout(0.25, name='dropout_4')(x)
    
    # Capas finales (Flatten -> Dense -> Drop -> Output Dense)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_5')(x)
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compilar el modelo (mantener la compilación aquí es opcional,
    # a menudo se compila después de crear el modelo)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Opcional: Imprimir resumen para verificar el conteo de capas (serializables)
    print("\nResumen del modelo creado (sin CBAM):")
    model.summary()
    # La suma de las capas listadas en summary + las capas dentro de custom Layers
    # determina el conteo total para serialización. Sin los CBAMs, este conteo
    # debería ser 10 (o un número que el sistema de carga interprete como 10).
    
    return model

# Funciones de entrenamiento y detección (mantienen su lógica, pero ahora usarán
# la versión modificada de create_cbam_4cnn)

# La función train_model_with_checkpoints y continue_training 
# necesitan los datos X_train, y_train, etc.
# Estas funciones probablemente se llamaban desde tu script de entrenamiento principal.
# Si quieres usarlas, necesitarías cargar los datos primero.
# Por simplicidad, si solo quieres que detect_emotion funcione cargando el modelo
# de 10 capas, la modificación clave ya está hecha en create_cbam_4cnn.

def train_model_with_checkpoints(model, X_train, y_train, X_val, y_val, 
                               epochs=100, batch_size=32, 
                               checkpoint_dir='checkpoints'):
     """
     Entrena el modelo con capacidad de guardar checkpoints por época.
     """
     if not os.path.exists(checkpoint_dir):
         os.makedirs(checkpoint_dir)

     checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5') # Mejor nombre para checkpoint
     csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'training_log.csv'), append=True)

     # Guardar solo los pesos, o el modelo completo?
     # Si el sistema de carga espera 10 capas, guardar el modelo completo
     # 'best_model.h5' con la nueva arquitectura es lo que necesitas.
     # ModelCheckpoint por defecto guarda el modelo completo si save_weights_only=False
     
     # Este checkpoint es diferente del que usaste en el script anterior ('best_model.h5')
     # Si quieres guardar el MEJOR modelo general, usa otro callback.
     # Si quieres checkpoints por época, usa este. Mantendré este tal cual estaba definido.
     checkpoint = ModelCheckpoint(
         checkpoint_path,
         monitor='val_accuracy',
         save_best_only=False, # Guarda cada época
         save_weights_only=True, # Solo guarda pesos para reanudar entrenamiento
         mode='max',
         verbose=1
     )

     # Callbacks adicionales (opcional, recomendado)
     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)


     print(f"Iniciando entrenamiento por {epochs} épocas...")
     history = model.fit(
         X_train, y_train,
         validation_data=(X_val, y_val),
         epochs=epochs,
         batch_size=batch_size,
         callbacks=[csv_logger, checkpoint, early_stopping, reduce_lr] # Usar callbacks
     )

     return history

def load_latest_checkpoint(model, checkpoint_dir='checkpoints'):
     """
     Carga los pesos del último checkpoint guardado.
     """
     if not os.path.exists(checkpoint_dir):
         print(f"Directorio de checkpoints no encontrado: {checkpoint_dir}")
         return 0

     # Buscar checkpoints (ahora con el nuevo formato de nombre si usas el checkpoint de arriba)
     checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_')]
     if not checkpoints:
         print("No se encontraron checkpoints en el directorio.")
         return 0

     # Obtener el número de la última época del checkpoint más reciente
     try:
         # Asume que el formato es model_epoch_XX_...h5
         latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2]))
         latest_epoch = int(latest_checkpoint.split('_')[2])
         latest_path = os.path.join(checkpoint_dir, latest_checkpoint)
         print(f"Cargando pesos del checkpoint más reciente: {latest_path}")
         model.load_weights(latest_path)
         print("Pesos cargados exitosamente.")
         return latest_epoch
     except Exception as e:
         print(f"Error al encontrar o cargar el último checkpoint: {str(e)}")
         print("Comenzando entrenamiento desde cero.")
         return 0


def continue_training(model, X_train, y_train, X_val, y_val, 
                     total_epochs=100, batch_size=32,
                     checkpoint_dir='checkpoints'):
     """
     Continúa el entrenamiento desde el último checkpoint hasta total_epochs.
     """
     last_epoch = load_latest_checkpoint(model, checkpoint_dir)

     if last_epoch >= total_epochs:
         print(f"El modelo ya ha sido entrenado por {last_epoch} épocas (igual o más que el total deseado de {total_epochs}).")
         return

     remaining_epochs = total_epochs - last_epoch
     print(f"Continuando entrenamiento. Época inicial: {last_epoch + 1}. Épocas restantes: {remaining_epochs}")

     # Entrenar las épocas restantes
     history = train_model_with_checkpoints(
         model, X_train, y_train, X_val, y_val,
         epochs=remaining_epochs, # Entrenar solo las épocas restantes
         batch_size=batch_size,
         checkpoint_dir=checkpoint_dir
     )

     # Guardar el modelo final después de completar todas las épocas deseadas
     final_model_path = 'final_trained_model.h5'
     model.save(final_model_path)
     print(f"Modelo final guardado como '{final_model_path}'")

     return history


# Esta función train_in_blocks parece una forma simplificada o alternativa
# a continue_training. Si tu flujo principal usa train_model_with_checkpoints/continue_training,
# quizás esta no es necesaria. Si la usas, asegúrate de que el load_model
# al inicio usa custom_objects={'CBAMBlock': CBAMBlock} si el best_model.h5
# pudo haber sido guardado con la versión vieja del modelo.
def train_in_blocks(X_train, y_train, X_val, y_val, epochs_per_block=5, total_blocks=20):
    """
    Entrena el modelo en bloques de épocas. NOTA: Este enfoque puede sobrescribir
    'best_model.h5' con un modelo menos preciso si save_best_only=False.
    Usar ModelCheckpoint con save_best_only=True en el fit directamente es más seguro.
    Mantengo la estructura, pero prefiero el enfoque con continue_training.
    """
    # Crear o cargar el modelo - Asegúrate de usar custom_objects si cargas modelos viejos
    try:
        print("Intentando cargar modelo 'best_model.h5'...")
        # Cargar con custom_objects por si el archivo fue creado con la version vieja (con CBAM)
        model = tf.keras.models.load_model('best_model.h5', custom_objects={'CBAMBlock': CBAMBlock})
        print("Modelo cargado exitosamente desde 'best_model.h5'")
        # Después de cargar un modelo viejo, la arquitectura está fijada.
        # Si quieres entrenar la nueva arquitectura (sin CBAM), debes CREAR un nuevo modelo.
        # Entonces, si la carga tiene éxito (es un modelo viejo de 14 capas), podrías decidir:
        # 1. Seguir entrenando el modelo viejo (de 14 capas) para mejorarlo, o
        # 2. Desechar el modelo viejo y crear uno nuevo (de 10 capas) para entrenar desde cero.
        # Dado el problema "expected 10 layers", la opción 2 es la que quieres para CORREGIR.
        print("Ignorando modelo cargado (probablemente de 14 capas) para entrenar un nuevo modelo de 10 capas.")
        model = create_cbam_4cnn(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
        print("Nuevo modelo de 10 capas creado.")

    except Exception as e:
        print(f"Error al cargar modelo 'best_model.h5': {e}. Creando nuevo modelo de 10 capas.")
        # Si la carga falla (porque no existe o es incompatible), crea el nuevo modelo
        model = create_cbam_4cnn(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
        print("Nuevo modelo de 10 capas creado.")

    # Configurar callback para guardar el mejor modelo general
    checkpoint = ModelCheckpoint(
        'best_model.h5', # Guarda el mejor modelo (arquitectura + pesos) aquí
        monitor='val_accuracy',
        save_best_only=True,
        mode='max', 
        verbose=1
    )

    # Callbacks adicionales (opcional, recomendado)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

    print(f"Iniciando entrenamiento por {epochs_per_block * total_blocks} épocas ({total_blocks} bloques de {epochs_per_block} épocas cada uno).")
    # Esta función entrena por 'epochs_per_block' épocas.
    # Si necesitas entrenar en bloques *serialmente* (guardar, parar, reanudar más tarde),
    # deberías usar la lógica de continue_training.
    # Esta implementación entrena todas las épocas a la vez, pero guarda el mejor modelo.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_per_block * total_blocks, # Entrenar el total de épocas
        batch_size=32,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # El mejor modelo ya fue guardado por el checkpoint.
    # Guardar el modelo final *al final del fit* también es una opción si no usaste save_best_only=True
    # model.save('final_model_at_end_of_fit.h5')

    return history


def detect_emotion():
    """Función para detectar emociones en tiempo real usando el modelo entrenado."""
    
    # Crear el modelo con la arquitectura que coincide con best_model.h5 (ahora de 10 capas)
    # Es CRUCIAL que esta creación use la MISMA función create_cbam_4cnn
    # modificada que se usó para entrenar el modelo guardado.
    print("Creando modelo con la arquitectura esperada (10 capas)...")
    model = create_cbam_4cnn(input_shape=(48, 48, 1), num_classes=len(EMOTIONS))
    
    # Cargar los pesos del modelo
    try:
        print("Intentando cargar pesos desde 'best_model.h5'...")
        # Al cargar pesos en un modelo YA construido, la arquitectura debe COINCIDIR
        # Si el archivo 'best_model.h5' fue guardado *con* la versión vieja (14 capas),
        # cargar solo pesos aquí podría FALLAR porque el modelo actual (10 capas) no coincide.
        # Si el archivo fue guardado *con* la versión nueva (10 capas), DEBERÍA funcionar.
        # Si el error de 10 vs 14 persistiera AQUÍ, significaría que best_model.h5
        # *aún* es la versión de 14 capas, o que tu sistema de carga espera algo diferente.
        # Asumimos que, después de entrenar con el código modificado, best_model.h5
        # contendrá el modelo de 10 capas.
        
        # Nota: load_weights no necesita custom_objects a menos que la *clase* de la capa
        # personalizada no esté definida en el entorno actual. Como CBAMBlock está definida,
        # no debería ser necesario aquí *para load_weights*. Para load_model (modelo completo), sí.
        model.load_weights('best_model.h5')
        print("Pesos del modelo cargados exitosamente desde 'best_model.h5'")
        
    except Exception as e:
        print(f"Error al cargar los pesos desde 'best_model.h5': {str(e)}")
        print("Usando modelo sin entrenar (inicializado aleatoriamente).")
        # Si la carga falla, el modelo usará los pesos aleatorios iniciales.
    
    # Inicializar la cámara y el detector de rostros
    cap = cv2.VideoCapture(0)
    # Usar cv2.data.haarcascades para encontrar el archivo
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        print(f"Error: Archivo de cascada no encontrado en {face_cascade_path}")
        cap.release()
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if not face_cascade.empty():
        print("Detector de rostros cargado exitosamente.")
    else:
         print("Error: No se pudo cargar el detector de rostros.")
         cap.release()
         return
    
    print("\nPresiona 'q' para salir...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de la cámara.")
            break
            
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        # Ajustar parámetros si es necesario (ej: scaleFactor, minNeighbors)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1, # Reducir para detectar rostros más pequeños/lejanos
            minNeighbors=5,  # Aumentar para reducir falsos positivos
            minSize=(30, 30) # Tamaño mínimo de un rostro
        )
        
        for (x, y, w, h) in faces:
            # Extraer el ROI (Región de Interés)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Preprocesar el ROI para la entrada del modelo
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0) # Añadir dimensión de batch
                roi_gray = np.expand_dims(roi_gray, axis=-1) # Añadir dimensión del canal (gris)
            except Exception as resize_error:
                 print(f"Error al redimensionar o preprocesar ROI: {resize_error}")
                 continue # Saltar este rostro si hay error

            # Predecir la emoción
            try:
                preds = model.predict(roi_gray, verbose=0)[0] # verbose=0 para no imprimir el progreso
                emotion_index = np.argmax(preds)
                emotion = EMOTIONS[emotion_index]
                confidence = preds[emotion_index] * 100 # Confianza en porcentaje
            except Exception as predict_error:
                print(f"Error durante la predicción: {predict_error}")
                emotion = "Error"
                confidence = 0

            # Dibujar el rectángulo y la emoción con confianza
            label = f"{emotion}: {confidence:.1f}%"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Ajustar posición del texto para que no se salga de la pantalla
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el frame
        cv2.imshow('Emotion Detection', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    print("Detección de emociones finalizada.")


# Bloque principal para ejecutar la detección si el script se ejecuta directamente
if __name__ == "__main__":
    # Configurar GPU si está disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU configurada con crecimiento de memoria.")
        except:
            print("Advertencia: GPU encontrada pero no se pudo configurar el crecimiento de memoria.")
    else:
        print("No se encontraron dispositivos GPU, usando CPU.")

    print("Iniciando detección de emociones...")
    detect_emotion()