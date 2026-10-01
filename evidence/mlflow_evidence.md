# MLflow Evidence

## Experiment

- Experimentname: `personality_type`
- Auswahlmetrik: `cv_f1_macro`
- Kreuzvalidierung: fünffache stratifizierte Kreuzvalidierung
- Testdaten wurden bei der Modellauswahl nicht verwendet.

## Modellvergleich

| Run | Modell | Beste Einstellung | CV-F1-Macro |
|---|---|---|---:|
| `LogisticRegression_tuned` | Logistische Regression | `C = 10.0` | 0,7683 |
| `RandomForest_tuned` | Random Forest | `max_depth = None`, `n_estimators = 100` | 0,7453 |

## Champion-Auswahl

Die logistische Regression wurde als Champion ausgewählt, weil sie beim mittleren Macro-F1-Wert der Kreuzvalidierung besser abschnitt als der Random Forest.

- Champion-Modell: Logistische Regression
- Dokumentierte Champion-Run-ID: `8fc4677111f24798aac65c1cac66c5ab`
- MLflow-Tag: `selection = champion`

## Abschließende Testbewertung

Erst nach der Modellauswahl wurde das Champion-Modell einmalig mit den zuvor zurückgehaltenen Testdaten bewertet.

| Kennzahl | Ergebnis |
|---|---:|
| Test-Accuracy | 0,8471 |
| Test-F1-Macro | 0,7545 |

Der Test-F1-Macro liegt nur geringfügig unter dem Kreuzvalidierungswert. Das spricht dafür, dass die Modellleistung auf den zurückgehaltenen Daten ähnlich ausfällt.

## Streamlit-App

Die Streamlit-App sucht im Experiment `personality_type` automatisch den neuesten Run mit dem Tag `selection = champion`. Anschließend lädt sie die vollständige gespeicherte Pipeline über die aktuelle Run-ID.

Bei einem neuen Modeling-Durchlauf erzeugt MLflow eine neue Run-ID. Durch die Suche nach dem Champion-Tag muss die Run-ID in `app.py` nicht manuell angepasst werden.