import pandas as pd

file_path2 = "C:/Users/LENOVO/Downloads/Data_CK/Vietnam_volume_filtered.csv"
df2 = pd.read_csv(file_path2, dtype=str, low_memory=False)

file_path3 = "C:/Users/LENOVO/Downloads/Data_CK/Vietnam_Price_filtered.csv"
df1 = pd.read_csv(file_path3, dtype=str, low_memory=False)

df1_melted = df1.melt(id_vars=["Name", "Code"], var_name="Date", value_name="Price")

df1_melted = df1_melted.dropna()

df1_melted["Date"] = pd.to_datetime(df1_melted["Date"])


df1_melted = df1_melted.drop(columns=["Name"])

df2_melted = df2.melt(id_vars=["Name", "Code"], var_name="Date", value_name="Volume")


df2_melted["Date"] = pd.to_datetime(df2_melted["Date"])
df2_melted = df2_melted.drop(columns=["Name"])

import pandas as pd

# Bước 1: Lấy danh sách ngày có trong df1
valid_dates = df1_melted["Date"].unique()

# Bước 2: Lọc df2 chỉ giữ lại những ngày có trong df1
df2_filtered = df2_melted[df2_melted["Date"].isin(valid_dates)].copy()

# Bước 3: Gộp df1 và df2 theo 'Code' và 'Date'
merged_df = pd.merge(df1_melted, df2_filtered, on=["Code", "Date"], how="left")

# Bước 4: Xóa dòng có Volume là NaN hoặc 0
merged_df = merged_df.dropna(subset=["Volume"])
merged_df = merged_df[merged_df["Volume"] != 0]

output_path = "C:/Users/LENOVO/Downloads/Data_CK/Merged_Data.csv"

merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"File đã được lưu tại: {output_path}")
