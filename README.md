# 🛵 Food Delivery Time Predictor — Streamlit App

A single-file Streamlit app (frontend + backend in one) that loads your trained
`xgb_best.pkl` + `standard_scaler.pkl` and predicts delivery time in minutes.

## Folder structure

```
food_delivery_app/
├── app.py                 # the Streamlit app (UI + inference)
├── requirements.txt       # Python dependencies
├── xgb_best.pkl            # your trained XGBoost model
├── standard_scaler.pkl     # your fitted StandardScaler
└── README.md
```

All four files **must stay in the same folder** — `app.py` loads the two
`.pkl` files using a path relative to itself.

## 1. Run it locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

It opens at `http://localhost:8501`.

> **Version note:** the model was pickled with whatever scikit-learn/xgboost
> version you trained it with (likely on Google Colab). If you get an
> "unpickle" or "attribute error" when the app loads the model, install the
> same major versions you trained with instead of the pinned ones in
> `requirements.txt` — check with `pip show scikit-learn xgboost` inside the
> Colab notebook.

## 2. Deploy for free — Streamlit Community Cloud

1. **Push this folder to a public GitHub repo** (e.g. a new repo
   `food-delivery-time-predictor`). Include `app.py`, `requirements.txt`,
   `xgb_best.pkl`, and `standard_scaler.pkl` — GitHub allows files this size
   without Git LFS.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with your GitHub account.
3. Click **"New app"**.
4. Select your repo, branch (`main`), and set **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit installs `requirements.txt` automatically and
   gives you a public URL like:
   `https://food-delivery-time-predictor.streamlit.app`
6. Add that link to your GitHub repo description and your portfolio site.

## 3. Alternative: Hugging Face Spaces

1. Create a new **Space** → SDK: **Streamlit**.
2. Upload the same 4 files.
3. Space auto-builds and deploys — you get a URL like
   `https://huggingface.co/spaces/<username>/food-delivery-time-predictor`.

## 4. How the prediction pipeline works (matches your notebook exactly)

- **Distance**: computed live from the 4 lat/lon inputs using the same
  haversine formula from the notebook.
- **Road_traffic_density**: ordinal-encoded (`Low=0, Medium=1, High=2, Jam=3`),
  same mapping used in training.
- **Type_of_order, Type_of_vehicle, Festival, City, Weatherconditions**:
  one-hot encoded to the exact same 21 columns, in the exact same order, that
  `X_train.columns` had after `pd.get_dummies(..., drop_first=True)`.
- The row is passed through `standard_scaler.pkl` (`.transform`, never
  `.fit_transform` — same as inference in your notebook), then into
  `xgb_best.pkl.predict()`.

If you ever retrain the model with different features, update
`FEATURE_COLUMNS` and `build_feature_row()` in `app.py` to match the new
`X_train.columns` output.

## 5. Model performance (from your notebook)

| Model | Test R² | Test RMSE (min) |
|---|---|---|
| Linear Regression | 0.57 | 6.21 |
| Decision Tree | 0.67 | 5.46 |
| Random Forest | 0.82 | 4.03 |
| **XGBoost (tuned)** | **0.83** | **3.91** |
