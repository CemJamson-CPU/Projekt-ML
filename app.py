from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st


# Projektpfad, MLflow-Datenbank und ausgewählter Champion-Run
PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_DIR / "mlflow.db"
EXPERIMENT_NAME = "personality_type"


# Die 19 Fragen und ihre Spaltennamen aus dem Trainingsdatensatz
QUESTIONS = {
    "N6": "I get upset easily.",
    "N8": "I have frequent mood swings.",
    "N7": "I change my mood a lot.",
    "N1": "I get stressed out easily.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "N3": "I worry about things.",
    "N5": "I am easily disturbed.",
    "E3": "I feel comfortable around people.",
    "N2": "I am relaxed most of the time.",
    "E5": "I start conversations.",
    "N4": "I seldom feel blue.",
    "E7": "I talk to a lot of different people at parties.",
    "E4": "I keep in the background.",
    "C4": "I make a mess of things.",
    "E1": "I am the life of the party.",
    "A4": "I sympathize with others' feelings.",
    "E10": "I am quiet around strangers.",
    "E9": "I don't mind being the center of attention.",
}


# Antwortskala aus dem Codebook
SCALE = {
    1: "Disagree",
    2: "Slightly disagree",
    3: "Neutral",
    4: "Slightly agree",
    5: "Agree",
}


# Aktuellen Champion-Run suchen und Pipeline nur einmal aus MLflow laden
@st.cache_resource
def load_champion_model():
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            "The MLflow experiment was not found. "
            "Run modeling.ipynb first."
        )

    champion_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.selection = 'champion'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if champion_runs.empty:
        raise RuntimeError(
            "No champion run was found. "
            "Run modeling.ipynb first."
        )

    champion_run_id = champion_runs.iloc[0]["run_id"]

    return mlflow.sklearn.load_model(
        f"runs:/{champion_run_id}/model"
    )


# Aufbau der Benutzeroberfläche
st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="🧠",
)

st.title("Personality Type Predictor")

st.write(
    "Answer all questions on a scale from 1 (Disagree) "
    "to 5 (Agree). The trained model then predicts a "
    "personality type."
)


# Alle Eingaben in einem Formular sammeln
with st.form("prediction_form"):
    answers = {}

    st.subheader("Personality questions")

    for column, question in QUESTIONS.items():
        answers[column] = st.select_slider(
            f"{column}: {question}",
            options=list(SCALE),
            value=3,
            format_func=lambda value: f"{value} – {SCALE[value]}",
        )

    st.subheader("Demographic information")

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=100,
        value=30,
        step=1,
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"],
    )

    hand = st.selectbox(
        "Handedness",
        ["Right", "Left", "Both"],
    )

    submitted = st.form_submit_button(
        "Predict personality type"
    )


# Eingaben an die vollständige Modell-Pipeline übergeben
if submitted:
    input_data = pd.DataFrame(
        [
            {
                **answers,
                "age": age,
                "gender": gender,
                "hand": hand,
            }
        ]
    )

    # Zahlenspalten wie im gespeicherten MLflow-Schema beschreiben
    numeric_columns = list(QUESTIONS) + ["age"]
    input_data[numeric_columns] = input_data[
        numeric_columns
    ].astype("float64")

    try:
        champion_model = load_champion_model()
        prediction = champion_model.predict(input_data)[0]

        st.success(
            f"Predicted personality type: {prediction}"
        )

        st.caption(
            "This prediction is part of a learning project "
            "and is not a diagnosis."
        )

    except Exception as error:
        st.error(
            "The saved model could not be loaded or used "
            "for the prediction."
        )
        st.exception(error)