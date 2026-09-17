# 🛵 Food Delivery Time Predictor — Streamlit App

A professional, single-file Streamlit web application that predicts food delivery times dynamically. The system handles live geospatial distance calculations and features robust preprocessing to ensure highly accurate, production-ready inferences.

---

## 🔗 Connect with Me

Professional profiles aur portfolio dekhne ke liye links par click karein:

[![LinkedIn](https://shields.io)](https://linkedin.com)
[![GitHub](https://shields.io)](https://github.com)
[![Kaggle](https://shields.io)](https://www.kaggle.com/datawithabxrehman)

---

## 🛠️ Built With

Project mein use hone wali core technologies:

![Python](https://shields.io)
![Streamlit](https://shields.io)
![XGBoost](https://shields.io)
![Scikit-Learn](https://shields.io)
![Pandas](https://shields.io)
![NumPy](https://shields.io)

---

## 📂 Folder Structure

GitHub repository ka clean setup is tarah hona chahiye:

```text
📂 food_delivery_app/
├── 📄 app.py                  # Streamlit application (UI + Inference pipeline)
├── 📄 requirements.txt        # Python libraries with explicit versions
├── 📊 model.json              # Trained XGBoost model (Native JSON format)
├── ⚙️ standard_scaler.pkl     # Fitted StandardScaler object
└── 📝 README.md               # Project documentation
```

> ⚠️ **Zaroori Baat:** Yeh charo files **ek hi directory** mein honi chahiye kyunki `app.py` in artifact files ko relative path se load karta hai.

---

## 🚀 1. Local Machine Par Run Kaise Karein

Aap niche diye gaye step-by-step terminal commands ko copy paste kar ke app locally run kar sakte hain:

```bash
# 1. Virtual Environment create karein (Recommended)
python -m venv venv
source venv/bin/activate      # For Windows: venv\Scripts\activate

# 2. Required dependencies install karein
pip install -r requirements.txt

# 3. Streamlit application ko launch karein
streamlit run app.py
```

Launch hone ke baad app local address `http://localhost:8501` par auto-open ho jayegi.

> 💡 **Model Format Note:** XGBoost model ko native `.json` format (`model.save_model("model.json")`) par save kiya gaya hai. Iska faida yeh hai ke environment badalne par `joblib` ya `pickle` ki tarah serialization errors (`input stream corrupted`) bilkul nahi aate.

---

## 🌐 2. Streamlit Community Cloud Par Free Deploy Karein

1. Apne main project folder ko public **GitHub repository** par push karein.
2. **[share.streamlit.io](https://share.streamlit.io)** par ja kar apne GitHub account se login karein.
3. **"New app"** ke button par click karein.
4. Apni repository, branch (`main`), aur main file path ko `app.py` par set karein.
5. **Deploy** button par click karein. Streamlit aapki dependencies cloud par khud install karega aur ek public URL generate kar dega:  
   `https://food-delivery-time-predictor.streamlit.app`

---

## 🤗 3. Hugging Face Spaces (Alternative Deployment)

1. Hugging Face par naya **Space** banayein aur SDK mein **Streamlit** ko select karein.
2. Apne project ki charo files (`app.py`, `requirements.txt`, `model.json`, `standard_scaler.pkl`) wahan upload kar dein.
3. Space background mein auto-build ho kar live ho jayega.

---

## ⚙️ 4. Prediction Pipeline Kaise Kaam Karta Hai?

Yeh inference script training notebook ke execution logic se 100% match karti hai:
- **Live Distance:** User ke input kiye gaye 4 Lat/Lon coordinates se distance **Haversine formula** ke zariye live calculate hota hai.
- **Traffic Density:** Ordinal encoding automatic background mein execute hoti hai (`Low=0, Medium=1, High=2, Jam=3`).
- **One-Hot Encoding:** Input inputs ko unhi exact 21 features ke layout mein binary mapping di jati hai jo training ke waqt `pd.get_dummies` ke baad bani thi.
- **Scaling & Output:** Structured features pehle `standard_scaler.pkl` ke zariye transform hote hain aur phir `model.load_model("model.json")` ke input ban kar aakhri prediction minutes mein generate karte hain.

---

## 📈 5. Model Performance Results

Training notebook ke mutabik perform hone wale sabhi models ka comparison:

| Model Evaluation | Test R² | Test RMSE (Minutes) |
| :--- | :---: | :---: |
| Linear Regression | 0.57 | 6.21 |
| Decision Tree | 0.67 | 5.46 |
| Random Forest | 0.82 | 4.03 |
| **XGBoost (Hyperparameter Tuned)** | **0.83** | **3.91** |
