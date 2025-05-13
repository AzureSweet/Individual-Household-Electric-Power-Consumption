import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib

# Đọc dữ liệu huấn luyện
df = pd.read_csv("train_data.csv", parse_dates=["Datetime"], index_col="Datetime")

df = df.dropna(subset=["Global_active_power"])  
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")  
df = df.dropna(subset=["Global_active_power"])  
# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[["Global_active_power"]])

# Huấn luyện HMM (3 trạng thái)
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
model.fit(scaled_data)

# Lưu mô hình và scaler
joblib.dump(model, "hmm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Thành công")

