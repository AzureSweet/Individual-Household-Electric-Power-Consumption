# check.py
import pandas as pd
import joblib
from particle_filter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Xử lý lỗi file
try:
    test_df = pd.read_csv("test_data_thursday.csv", parse_dates=["Datetime"], index_col="Datetime", na_values="?")
    hmm_model = joblib.load("hmm_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    exit(1)

# Làm sạch dữ liệu
test_df = test_df.dropna(subset=["Global_active_power"])
test_df["Global_active_power"] = pd.to_numeric(test_df["Global_active_power"], errors="coerce")
test_df = test_df.dropna(subset=["Global_active_power"])

if test_df.empty:
    exit(1)

# Khởi tạo Particle Filter
pf = ParticleFilter(n_particles=500, n_states=3)

true_vals = []
predicted_vals = []

for obs in test_df["Global_active_power"]:
    pf.update(obs, hmm_model, scaler)
    estimate = pf.estimate()
    most_likely_state = np.argmax(estimate)
    
    # Chuyển trạng thái thành giá trị Global_active_power
    scaled_mean = hmm_model.means_[most_likely_state]
    predicted_val = scaler.inverse_transform([scaled_mean])[0]
    
    true_vals.append(obs)
    predicted_vals.append(predicted_val)

# Trực quan hóa
plt.figure(figsize=(12, 4))
plt.plot(test_df.index, true_vals, label="Thực tế")
plt.plot(test_df.index, predicted_vals, label="Dự báo (Particle Filter)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_result.png")
plt.show()

# Tính lỗi
rmse = np.sqrt(mean_squared_error(true_vals, predicted_vals))  # Tính RMSE thủ công
mae = mean_absolute_error(true_vals, predicted_vals)

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")