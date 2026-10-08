from pathlib import Path
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "comentarios.csv"
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"
CM_PATH = ROOT / "outputs" / "confusion_matrix.png"
PRED_PATH = ROOT / "outputs" / "predicciones_test.csv"


def limpiar_texto(texto: str) -> str:
    """Normaliza texto en español sin eliminar negaciones."""
    texto = str(texto).lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)
    texto = re.sub(r"#(\w+)", r"\1", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    requeridas = {"comentario", "tipo"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    df = df.dropna(subset=["comentario", "tipo"]).copy()
    df["comentario_limpio"] = df["comentario"].map(limpiar_texto)

    # Evita fuga de información si el mismo comentario aparece repetido.
    df = df.drop_duplicates(subset=["comentario_limpio"]).reset_index(drop=True)
    return df


def main() -> None:
    df = cargar_datos()

    print("Registros únicos utilizados:", len(df))
    print("\nDistribución de clases:")
    print(df["tipo"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["comentario_limpio"],
        df["tipo"],
        test_size=0.20,
        random_state=42,
        stratify=df["tipo"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "modelo",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nReporte de clasificación:\n")
    print(classification_report(y_test, pred, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    resultados = pd.DataFrame(
        {
            "comentario": X_test.values,
            "real": y_test.values,
            "prediccion": pred,
        }
    )
    resultados.to_csv(PRED_PATH, index=False)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        cmap=None,
        xticks_rotation=45,
    )
    disp.ax_.set_title("Matriz de confusión - Análisis de sentimiento")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    ejemplos = [
        "La camioneta está excelente, me encantó",
        "No me gustó para nada, fue una mala experiencia",
        "Quiero más información sobre el crédito",
    ]
    print("\nPredicciones de ejemplo:")
    for texto, etiqueta in zip(ejemplos, pipeline.predict(ejemplos)):
        print(f"- {texto!r} -> {etiqueta}")

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Matriz de confusión guardada en: {CM_PATH}")
    print(f"Predicciones de test guardadas en: {PRED_PATH}")


if __name__ == "__main__":
    main()
