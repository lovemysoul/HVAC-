# ❄️ HVAC Failure Predictor

### *Because nobody wants their air conditioner to die in the middle of an Indian summer.*

A machine learning project that looks at sensor readings from an HVAC unit and says: **"Relax, it's fine"** or **"Bro, call a technician."**

Built with Python, scikit-learn and Streamlit. Powered by chai and mild panic. ☕

---

## 🎬 Live Demo

👉 **[Try it here](PASTE-YOUR-STREAMLIT-LINK-HERE)** *(if the link is dead, the app is probably sleeping, give it a minute, we all need naps)*

---

## 🧐 What does it do?

You type in 6 numbers:

| Sensor | What it means |
|---|---|
| 🌡️ Temperature (°C) | How hot the machine is |
| 💧 Humidity (%) | How sweaty the air is |
| 📳 Vibration (mm/s) | How much the machine is shaking (the *important* one) |
| 🎈 Pressure (kPa) | How stressed the pipes are |
| ⚡ Energy (kW) | How much electricity it's gulping |
| ⏱️ Runtime (hours) | How long it's been working without a holiday |

The app gives you a **failure probability** and a verdict: ✅ *Looks normal* or ⚠️ *High risk, schedule an inspection.*

You can choose between two models: **Random Forest** and **Logistic Regression**. They don't always agree. Just like family at a wedding.

---

## 📦 The Data

- **4,000 hourly readings** from an HVAC unit (`HVAC_Sensor_Data.csv`)
- Only **169 failures (4.2%)**. Failures are rare, and that makes everything harder.
- **2% missing values** in each of the 5 sensor columns (sensors blinking off like a bad WiFi connection)

---

## 🛠️ What I Did (the honest journey)

### 1. Missing values 🕳️
Filled with the **median**. Why not the mean? Because outliers drag the mean around like a toddler with a shopping cart. The median doesn't care.

### 2. Outliers 🚨
I capped them with the IQR method for the before/after plots, then discovered something spicy:

> Rows with vibration outliers fail **23.8%** of the time. The overall failure rate is **4.2%**.

Those "outliers" aren't noise. They're the machine **screaming for help**. So the models were trained on the **uncapped** data. Capping them would be like taping over the check-engine light. 🔧

### 3. The Bug That Deleted All My Failures 💀
My first capping loop included the `failure` column itself. Since most rows are 0, the IQR was 0, so every single value got clipped to 0. **All 169 failures vanished.** The model would have had nothing to learn.

**Lesson:** never cap your target column. And always `print(df["failure"].value_counts())` after cleaning.

### 4. The Model That Gave Up 😴
My first random forest predicted **"no failure" for every single row**. Accuracy: **96%**. Useful predictions: **zero**.

A model that always says "no failure" scores 95.75% here. Accuracy is a trap on imbalanced data. 🪤

The fix: the forest never got above **48%** confidence, so the default 0.5 threshold never fired. Lowering the threshold to **0.10** made it speak up.

### 5. Feature Engineering? 🧪
I tried time features, interactions, rolling averages and a high-vibration flag with 5-fold cross-validation. **None of them beat the original 6 features.** Sometimes the honest answer is "it didn't help."

---

## 📊 Results

| | 🌲 Random Forest | 📈 Logistic Regression |
|---|---|---|
| **AUC** (fair comparison) | **0.777** | 0.752 |
| Threshold | 0.10 | 0.50 |
| Failures caught (recall) | 14 / 34 (41%) | 21 / 34 (62%) |
| False alarms | 81 | 217 |
| Precision | 15% | 9% |
| Accuracy | 87% | 71% |

**How to read this:**
- Both models beat a coin flip (AUC 0.5) comfortably.
- Logistic Regression catches more failures but cries wolf a lot more. Random Forest is calmer but misses more.
- Accuracy looks lower than the "always say no" model on purpose. That model is cheating, I'm not.

### 🏆 Feature Importance (Random Forest)

1. 📳 **Vibration: 0.28** (the clear champion)
2. 🎈 Pressure: 0.17
3. 💧 Humidity: 0.14
4. 🌡️ Temperature: 0.14
5. ⏱️ Runtime: 0.14
6. ⚡ Energy: 0.13

Translation: **if it's shaking, worry.**

---

## 🚀 Run It Yourself

```bash
git clone https://github.com/lovemysoul/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

### 🧪 Sample inputs to try

| | Healthy machine | Shaky machine |
|---|---|---|
| Temperature | 24.0 | 27.0 |
| Humidity | 51.0 | 58.0 |
| Vibration | 0.65 | 2.5 |
| Pressure | 101.0 | 104.0 |
| Energy | 18.0 | 21.0 |
| Runtime | 1600 | 2500 |
| **Expected** | ✅ Looks normal | ⚠️ High risk |

**Fun experiment:** keep the healthy values and slowly raise only the vibration. Watch the probability climb. The model really, *really* cares about shaking.

---

## 📁 Project Structure

```
hvac_app/
├── app.py              # Streamlit app
├── rf_model.pkl        # Trained random forest
├── lr_model.pkl        # Trained logistic regression
├── scaler.pkl          # StandardScaler (used by logistic regression only)
├── config.pkl          # Column order + thresholds + medians
├── requirements.txt
└── README.md           # You are here 👋
```

---

## ⚠️ Honest Limitations

- **Precision is low (9% to 15%).** Failures overlap heavily with normal operation, so many warnings will be false alarms. Treat a warning as *"go inspect"*, not *"it's definitely broken."*
- **Small data.** Only 169 failures to learn from.
- **Median imputation was done before the train/test split**, which leaks a tiny bit of test information. With only 2% missing it barely matters, but I'm mentioning it because honesty is cool.
- **This is a learning project**, not something to run an actual building on. Please don't fire your maintenance team. 🙏

---

## 🧰 Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn` · `Streamlit` · `joblib`

---

## 👨‍💻 Author

**Abhinav K** (Abhi)
BSc CS (AI & ML)

🐙 GitHub: [github.com/lovemysoul](https://github.com/lovemysoul)

If this project helped you, drop a ⭐. It costs nothing and makes me irrationally happy.

---

*Made with curiosity, a lot of debugging, and exactly one accidental deletion of every failure in the dataset.* 😅
