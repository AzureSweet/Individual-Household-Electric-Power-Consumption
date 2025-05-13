import pandas as pd

# Đọc dữ liệu đã xử lý
df = pd.read_csv("processed_power_data.csv", parse_dates=["Datetime"], index_col="Datetime")

# Tạo đặc trưng ngày trong tuần (0 = Monday, 6 = Sunday)
df["weekday"] = df.index.weekday

# Tạo đặc trưng: IsThursday (để chọn các ngày thứ Năm cho kiểm thử hoặc dự báo)
df["is_thursday"] = (df["weekday"] == 3).astype(int)

# Chia tập huấn luyện và kiểm thử
train_df = df[df.index.year < 2010]
test_df = df[(df.index.year == 2010) & (df["is_thursday"] == 1)]

# Lưu tập train và test
train_df.to_csv("train_data.csv")
test_df.to_csv("test_data_thursday.csv")

print("Thành công")