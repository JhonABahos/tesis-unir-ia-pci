# modules/gpt4all_report.py

from datetime import datetime
from config import TEMPERATURE, MAX_TOKENS
from gpt4all import GPT4All

# Variable global para definir el modelo que se usará
# MODEL_NAME = "gpt4all-falcon-newbpe-q4_0.gguf" # menos preciso en español
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf" # más preciso pero más requerimiento de hardware
# MODEL_NAME = "wizardlm-13b-v1.2.Q4_0.gguf" # más preciso pero aún más requerimiento de hardware (no traduce a español)

# Inicializa el modelo GPT4All usando la variable MODEL_NAME
gpt = GPT4All(model_name=MODEL_NAME,
              model_path="D:/Proyectos_BC/ia_tesis_pci/models",
              allow_download=False)


def medir_tiempo_generacion(func):
    """
    Decorador para medir el tiempo de ejecución de las funciones de generación y traducción.
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        resultado = func(*args, **kwargs)
        end_time = datetime.now()

        elapsed_time = end_time - start_time
        seconds = elapsed_time.total_seconds()
        minutes, seconds_only = divmod(seconds, 60)

        resultado += (f"\n\n(Execution time: {int(minutes)} minutes and {seconds_only:.2f} seconds | "
                      f"Total: {seconds:.2f} seconds)")
        return resultado
    return wrapper


@medir_tiempo_generacion
def generar_reporte_pavimento(pci, damage):
    """
    Genera un reporte técnico en inglés sobre el estado del pavimento.
    """
    prompt = (
        f"Write a technical report about the pavement condition in English. "
        f"The pavement has a PCI of {pci} and presents the following damage: {damage}. "
        f"Include detailed annotations and maintenance recommendations."
    )
    respuesta_ingles = gpt.generate(prompt, max_tokens=MAX_TOKENS, temp=TEMPERATURE)
    return respuesta_ingles


@medir_tiempo_generacion
def traducir_a_espanol(texto_en_ingles):
    """
    Traduce un texto técnico del inglés al español.
    """
    prompt_traduccion = (
        f"Traduce al español el siguiente informe técnico:\n\n"
        f"{texto_en_ingles}"
    )
    respuesta_espanol = gpt.generate(prompt_traduccion, max_tokens=MAX_TOKENS, temp=TEMPERATURE)
    return respuesta_espanol


if __name__ == "__main__":
    # Datos de prueba
    pci = 90
    damage = "longitudinal cracks and transverse cracks on the pavement"

    # Generar el reporte en inglés
    print(f"Using model: {MODEL_NAME}")
    print("Generating report in English...")
    reporte_ingles = generar_reporte_pavimento(pci, damage)
    print("\n--- Report in English ---")
    print(reporte_ingles)

    # Traducir el reporte al español
    print("\nTranslating report to Spanish...")
    reporte_espanol = traducir_a_espanol(reporte_ingles)
    print("\n--- Reporte en español ---")
    print(reporte_espanol)
    print("\n--- Fin de las respuestas ---")
