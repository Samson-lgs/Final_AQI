import os
import pandas as pd
import psycopg2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------- DATABASE CONNECTION ----------------------
DB_HOST = "ep-patient-breeze-adt1xy0o-pooler.c-2.us-east-1.aws.neon.tech"
DB_NAME = "neondb"
DB_USER = "neondb_owner"
DB_PASSWORD = "npg_1yEJuba0dzxi"

conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    sslmode="require"
)

query = "SELECT city, temperature, humidity, pressure FROM openweather_data;"
df = pd.read_sql_query(query, conn)
conn.close()

print(f"✅ Data fetched: {len(df)} rows")

# ---------------------- DATA PREPROCESSING ----------------------
df = df.dropna()
if len(df) < 10:
    raise ValueError("❌ Not enough data to train models. Add more weather data first.")

# Simulate AQI temporarily (for demonstration)
df["AQI"] = (df["temperature"] * 2) + (df["humidity"] * 0.5) + (df["pressure"] * 0.01)

X = df[["temperature", "humidity", "pressure"]]
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------- CREATE MODELS FOLDER ----------------------
os.makedirs("models", exist_ok=True)

# ---------------------- MODEL 1: LINEAR REGRESSION ----------------------
print("\n📘 Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
joblib.dump(lr, "models/linear_regression.pkl")

# ---------------------- MODEL 2: RANDOM FOREST ----------------------
print("🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
joblib.dump(rf, "models/random_forest.pkl")

# ---------------------- MODEL 3: XGBOOST ----------------------
print("⚡ Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
joblib.dump(xgb, "models/xgboost_model.pkl")

# ---------------------- MODEL 4: LSTM ----------------------
print("🧠 Training LSTM...")
X_lstm = np.array(X)
y_lstm = np.array(y)
X_lstm = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))

model_lstm = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_lstm.shape[2])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model_lstm.fit(X_lstm, y_lstm, epochs=30, batch_size=8, verbose=0, callbacks=[es])

X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
lstm_pred = model_lstm.predict(X_test_lstm, verbose=0)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
model_lstm.save("models/lstm_model.h5")

# ---------------------- RESULTS ----------------------
print("\n📊 Model Performance:")
print(f"Linear Regression RMSE: {lr_rmse:.3f}")
print(f"Random Forest RMSE: {rf_rmse:.3f}")
print(f"XGBoost RMSE: {xgb_rmse:.3f}")
print(f"LSTM RMSE: {lstm_rmse:.3f}")

# ---------------------- PREDICT WITH ALL MODELS ----------------------
sample = np.array([[30, 60, 1013]])  # Example input
print("\n🔮 Predictions for sample input (Temp=30°C, Humidity=60%, Pressure=1013hPa):")
print("Linear Regression:", lr.predict(sample)[0])
print("Random Forest:", rf.predict(sample)[0])
print("XGBoost:", xgb.predict(sample)[0])

sample_lstm = sample.reshape((1, 1, 3))
print("LSTM:", model_lstm.predict(sample_lstm, verbose=0)[0][0])

print("\n✅ All models trained and saved successfully in the 'models/' folder!")
