# Detección de Emociones en Tiempo Real

Este programa implementa un sistema de detección de emociones en tiempo real utilizando una red neuronal convolucional con módulos de atención CBAM (CBAM-4CNN) y el dataset FER2013+.

## Requisitos

- Python 3.7 o superior
- Cámara web
- Las dependencias listadas en `requirements.txt`

## Instalación

1. Clonar este repositorio
2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar el programa:

```bash
python emotion_detection.py
```

- El programa iniciará la cámara web y comenzará a detectar emociones en tiempo real
- Presiona 'q' para salir del programa

## Características

- Detección de 7 emociones: Enojo, Disgusto, Miedo, Feliz, Triste, Sorpresa y Neutral
- Interfaz visual en tiempo real
- Utiliza el módulo de atención CBAM para mejorar la precisión
- Basado en la arquitectura CBAM-4CNN

## Notas

- Asegúrate de tener buena iluminación para mejores resultados
- El modelo necesita ser entrenado con el dataset FER2013+ antes de su uso
- Se recomienda usar una GPU para mejor rendimiento 