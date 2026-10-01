# Personality Type Predictor

Dieses Machine-Learning-Projekt sagt anhand von 19 Persönlichkeitsfragen sowie Alter, Geschlecht und Händigkeit einen Persönlichkeitstyp voraus.

Das Projekt umfasst die Datenanalyse, Datenbereinigung, Modellierung, Modellverwaltung mit MLflow und eine interaktive Streamlit-App.

## Projektaufbau

```text
Projekt-ML/
├── data/
│   ├── data.csv
│   ├── data_processed.csv
│   └── codebook.txt
├── eda.ipynb
├── modeling.ipynb
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Lokale Dateien wie `.venv`, `mlflow.db` und `mlruns` werden nicht in GitHub gespeichert.

## Vorgehensweise

### 1. Explorative Datenanalyse

Im Notebook `eda.ipynb` werden:

- die Datenqualität geprüft,
- relevante Werte bereinigt,
- fehlende Angaben behandelt,
- und die bereinigten Daten als `data/data_processed.csv` gespeichert.

### 2. Preprocessing und Modeling

Im Notebook `modeling.ipynb` werden:

- Eingabespalten und Zielspalte getrennt,
- die Daten in 80 % Trainingsdaten und 20 % Testdaten aufgeteilt,
- fehlende numerische Werte mit dem Median ersetzt,
- numerische Werte mit dem `StandardScaler` skaliert,
- fehlende kategorische Werte als `Unknown` behandelt,
- kategorische Werte mit dem `OneHotEncoder` umgewandelt,
- und alle Schritte in vollständigen Modell-Pipelines verbunden.

### 3. Modellvergleich

Verglichen werden:

- logistische Regression
- Random Forest

Die Modelle werden zunächst mit einer fünffachen stratifizierten Kreuzvalidierung innerhalb der Trainingsdaten geprüft. Anschließend werden ausgewählte Einstellungen mit `GridSearchCV` abgestimmt.

Als Auswahlkennzahl wird der Macro-F1-Wert verwendet.

## Ergebnisse

| Modell | CV-F1-Macro |
|---|---:|
| Logistische Regression | 0,7683 |
| Random Forest | 0,7453 |

Die logistische Regression wurde als Champion-Modell ausgewählt.

Abschließende Bewertung auf den zuvor unbenutzten Testdaten:

| Kennzahl | Ergebnis |
|---|---:|
| Test-Accuracy | 0,8471 |
| Test-F1-Macro | 0,7545 |

## MLflow

Die abgestimmten Modelle, Einstellungen und Kennzahlen werden lokal in MLflow gespeichert.

Das ausgewählte Modell erhält den Tag:

```text
selection = champion
```

### Champion-Evidence

Als Auswahlmetrik wurde der mittlere Macro-F1-Wert der Kreuzvalidierung verwendet.

- Champion-Modell: Logistische Regression
- CV-F1-Macro: 0,7683
- Test-Accuracy: 0,8471
- Test-F1-Macro: 0,7545
- Dokumentierte Champion-Run-ID: `8fc4677111f24798aac65c1cac66c5ab`

Diese Run-ID gehört zum dokumentierten Modeling-Durchlauf des Hauptprojekts. Bei einem neuen vollständigen Modeling-Durchlauf erzeugt MLflow eine neue Run-ID. Die Streamlit-App sucht deshalb automatisch den neuesten mit `selection = champion` markierten Run und lädt dessen Pipeline über die aktuelle Run-ID.


MLflow kann im Projektordner folgendermaßen gestartet werden:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Danach ist MLflow unter folgender Adresse erreichbar:

```text
http://127.0.0.1:5000
```

## Installation

Repository klonen:

```bash
git clone https://github.com/CemJamson-CPU/Projekt-ML.git
cd Projekt-ML
```

Virtuelle Umgebung erstellen und aktivieren:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Abhängigkeiten installieren:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Projekt ausführen

Für eine vollständige Reproduktion werden die Dateien in dieser Reihenfolge ausgeführt:

1. `eda.ipynb` öffnen und vollständig mit **Run All** ausführen.
2. `modeling.ipynb` öffnen und vollständig mit **Run All** ausführen.
3. Die Streamlit-App starten:

```bash
python -m streamlit run app.py
```

Die App ist anschließend normalerweise unter folgender Adresse erreichbar:

```text
http://localhost:8501
```

Das Modeling-Notebook muss vor der App ausgeführt worden sein, da es die lokale MLflow-Datenbank, die Modellläufe und den Champion-Run erzeugt.

## Streamlit-App

Die App enthält Eingabefelder für:

- 19 Persönlichkeitsfragen auf einer Skala von 1 bis 5,
- Alter,
- Geschlecht,
- Händigkeit.

Nach dem Absenden werden die Eingaben an die vollständige Champion-Pipeline übergeben und der vorhergesagte Persönlichkeitstyp wird angezeigt.

## Hinweis

## Zielvariable und Klassen

Die vorherzusagende Zielvariable heißt `target`.

Im verwendeten Datensatz kommen folgende Klassen vor:

- Moderate
- Resilient
- Overcontroller
- Undercontroller

Das Codebook nennt zusätzlich die Klasse `Reserved`. Diese Klasse kommt im bereitgestellten Datensatz jedoch nicht vor und kann deshalb vom trainierten Modell nicht erlernt oder vorhergesagt werden.

## Bekannte Limitationen

- Die Zielvariable wurde aus Antworten des Persönlichkeitsfragebogens abgeleitet, die teilweise gleichzeitig als Eingaben des Modells dienen. Das Modell bildet daher vor allem die im Datensatz verwendete Zuordnungslogik nach.
- Die Ergebnisse wurden nur mit dem vorliegenden Datensatz überprüft. Eine externe Validierung mit unabhängigen Daten wurde nicht durchgeführt.
- Die im Codebook genannte Klasse `Reserved` fehlt im Datensatz.
- Die Vorhersage ist keine psychologische oder medizinische Diagnose.
- Die lokal erzeugten MLflow-Daten werden nicht bei GitHub gespeichert. Nach einem frischen Clone muss deshalb zuerst `modeling.ipynb` ausgeführt werden.

## Sinnvolle nächste Schritte

- Das Modell mit einem unabhängigen Datensatz validieren.
- Die fehlende Klasse `Reserved` fachlich und datenbezogen untersuchen.
- Weitere Modelle oder Einstellungen nur dann prüfen, wenn sie einen begründeten zusätzlichen Erkenntnisgewinn versprechen.
- Die Streamlit-App bereitstellen, falls für die Abgabe ein öffentlicher Deployment-Link verlangt wird.

Dieses Projekt wurde zu Lernzwecken erstellt. Die ausgegebene Vorhersage ist keine psychologische oder medizinische Diagnose.