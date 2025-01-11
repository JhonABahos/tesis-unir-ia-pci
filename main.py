#main.py

from modules.config import DATA_DIR
from modules.preprocessing import process_all_images
from modules.training_pipeline import train_model
from modules.inference import evaluate_example

def main():
    # 1. Procesar imágenes: segmentación y generación de máscaras
    print("Procesando imágenes para segmentación y generación de máscaras...")
    process_all_images(DATA_DIR)
    print("Procesamiento de imágenes completado.")

    # 2. Entrenar el modelo
    print("Iniciando el entrenamiento del modelo...")
    model = train_model(DATA_DIR)
    print("Entrenamiento completado.")

    # 3. Evaluar un ejemplo y calcular el PCI
    print("Calculando el PCI de un ejemplo...")
    evaluate_example(model, DATA_DIR)
    print("Ejecución completada.")

if __name__ == "__main__":
    main()
