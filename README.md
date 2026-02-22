# Columbus Crew xG Dashboard — 2025 MLS Season

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=flat-square&logo=chart.js&logoColor=white)](https://chartjs.org)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-222222?style=flat-square&logo=github&logoColor=white)](https://jakefraser04.github.io/Crew2025xG/)

**[→ View Live Dashboard](https://jakefraser04.github.io/Crew2025xG/)**

---

## The Question

The Columbus Crew entered 2025 without Cucho Hernández — the reigning MLS Cup MVP — and with
Wilfried Nancy entering his third and final season as head coach. Could a reshuffled squad
built around Diego Rossi hold its own in the Eastern Conference?

And more importantly: **did their final record actually reflect how they played?**

Expected Goals (xG) cuts through scorelines and tells you which results a team *earned*
and which they *got away with*. This project builds an xG model from scratch and applies
it to every Columbus Crew match in the 2025 MLS season to answer that question.

---

## What This Project Does

1. **Trains an xG model** using logistic regression on StatsBomb open data — thousands of
   labeled shots with location, technique, pressure, and outcome
2. **Engineers features** from raw shot coordinates: distance to goal, shot angle, body part,
   pressure, and play pattern
3. **Evaluates the model** with ROC-AUC, log loss, and a calibration curve
4. **Pulls real 2025 Columbus Crew match data** via the American Soccer Analysis API
5. **Applies the model** to generate match-by-match xG for and against
6. **Visualizes the season** — cumulative xG trend, match differential, rolling form,
   home/away breakdown, and a shot quality heatmap
7. **Serves everything** as an interactive dashboard on GitHub Pages

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | Python · pandas · NumPy · scikit-learn · joblib |
| Visualization | matplotlib · mplsoccer · seaborn |
| Data | StatsBomb open data · American Soccer Analysis API (`itscalledsoccer`) |
| Frontend | Vanilla JavaScript · Chart.js · HTML/CSS |
| Hosting | GitHub Pages |

---

## Project Structure

```
Crew2025xG/
├── notebooks/
│   ├── xg_model.ipynb            # Train logistic regression xG model
│   └── crew_xg_analysis_v2.ipynb # Apply model to 2025 Crew data
├── models/
│   ├── xg_model.pkl              # Saved trained model
│   └── model_meta.json           # AUC, log loss, training info
├── assets/
│   ├── xg_trend.png
│   ├── xg_differential.png
│   ├── rolling_xg_form.png
│   ├── home_away_xg.png
│   └── xg_heatmap.png
├── dashboard/
│   ├── index.html                # Interactive dashboard
│   └── data/
│       ├── match_xg.json         # Match-by-match xG data
│       └── season_summary.json   # Season headline stats
├── index.html                    # GitHub Pages entry point
└── requirements.txt
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/jakefraser04/Crew2025xG.git
cd Crew2025xG
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the xG model**

Open and run `notebooks/xg_model.ipynb` from top to bottom.
This pulls StatsBomb open data, trains the logistic regression model,
and saves it to `models/xg_model.pkl`.

> The StatsBomb data pull takes ~5 minutes on first run.

**4. Run the Crew analysis**

Open and run `notebooks/crew_xg_analysis_v2.ipynb`.
This pulls 2025 Columbus Crew match data via the ASA API,
applies the trained model, and exports JSON files for the dashboard.

**5. View the dashboard**
```bash
python -m http.server 8000
```
Then open `http://localhost:8000/dashboard/` in your browser.

> A local server is required because the dashboard loads JSON via `fetch()`.

---

## About xG

Expected Goals (xG) is a probability metric: given the location, angle, technique,
and context of a shot, what is the likelihood it results in a goal?

This model uses logistic regression trained on StatsBomb shot event data with the
following features:

| Feature | Description |
|---|---|
| Distance | Euclidean distance from shot location to goal center |
| Angle | Angle between shot location and goalposts (law of cosines) |
| Is Header | Whether the shot was taken with the head |
| Under Pressure | Whether the shooter was under defensive pressure |
| Is Open Play | Whether the shot came from open play vs. set piece |
| Distance² | Non-linear distance term to capture steep drop-off at range |

A shot with xG = 0.20 means shots taken from that location and context
score approximately 20% of the time historically.

---

## Data Sources

- **StatsBomb Open Data** — Shot event data for model training
  ([github.com/statsbomb/open-data](https://github.com/statsbomb/open-data))
- **American Soccer Analysis** — 2025 MLS match xG data
  ([americansocceranalysis.com](https://www.americansocceranalysis.com))

---

## Author

**Jake Fraser** · [github.com/jakefraser04](https://github.com/jakefraser04)
