from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe el modelo. Ejecuta primero: python src/train_sentiment.py"
        )

    texto = " ".join(sys.argv[1:]).strip()
    if not texto:
        texto = input("Escribe un comentario: ").strip()

    modelo = joblib.load(MODEL_PATH)
    prediccion = modelo.predict([texto])[0]
    probabilidades = modelo.predict_proba([texto])[0]

    print(f"\nComentario: {texto}")
    print(f"Predicción: {prediccion}")
    print("Probabilidades:")
    for clase, prob in zip(modelo.classes_, probabilidades):
        print(f"  {clase}: {prob:.2%}")


if __name__ == "__main__":
    main()
