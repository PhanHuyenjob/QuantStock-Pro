import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from openai import OpenAI
from tabulate import tabulate
# Constants
FILE_PATH = "C:/Users/LENOVO/Downloads/Data_CK/Merged_Data.csv"
DATA_DIR = "C:/Users/LENOVO/Downloads/Data_CK/data_GD_csv_cleaned_2"

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, dtype=str, low_memory=False)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    return data

# Filter data based on ticker and date range
def filter_data(data, ticker, start_date, end_date):
    filtered_data = data[(data["Code"] == ticker) & (data["Date"].between(start_date, end_date))]
    filtered_data = filtered_data.dropna(subset=["Price", "Volume"])
    filtered_data = filtered_data[filtered_data["Volume"] > 0]
    return filtered_data

# Extract date from filename
def extract_date_from_filename(filename):
    date_str = filename.split("_")[-1].split(".")[0]
    return datetime.strptime(date_str, "%Y%m%d")

def display_fundamental_data(ticker, start_date, end_date):
    all_files = os.listdir(DATA_DIR)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    filtered_files = []
    for file in csv_files:
        file_date = extract_date_from_filename(file)
        if start_date <= file_date <= end_date:
            filtered_files.append(file)
    
    if not filtered_files:
        st.warning("⚠ No data found for the selected stock and date range.")
        return
    
    sectors = set()
    for file in filtered_files:
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path, dtype=str, low_memory=False, encoding='utf-8-sig')
        if 'Ngành' in df.columns:
            sectors.update(df['Ngành'].unique())
    
    if not sectors:
        st.warning("⚠ Không có dữ liệu ngành trong các file đã chọn.")
        return
    
    selected_sector = st.selectbox("Chọn ngành", list(sectors))
    
    columns_to_plot = [
        "Cá nhân Khớp Ròng", "Tổ chức trong nước Khớp Ròng", "Tự doanh Khớp Ròng", "Nước ngoài Khớp Ròng",
        "Cá nhân Thỏa thuận Ròng", "Tổ chức trong nước Thỏa thuận Ròng", "Tự doanh Thỏa thuận Ròng", "Nước ngoài Thỏa thuận Ròng",
        "Cá nhân Tổng GT Ròng", "Tổ chức trong nước Tổng GT Ròng", "Tự doanh Tổng GT Ròng", "Nước ngoài Tổng GT Ròng"
    ]
    selected_columns = st.multiselect("Chọn các đường line muốn hiển thị", columns_to_plot, default=columns_to_plot[:4])
    
    plot_data = pd.DataFrame()
    
    for file in filtered_files:
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(file_path, dtype=str, low_memory=False, encoding='utf-8-sig')
        
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        
        if 'Ngành' in df.columns:
            sector_data = df[df['Ngành'] == selected_sector]
            if not sector_data.empty:
                file_date = extract_date_from_filename(file)
                sector_data['Date'] = file_date
                plot_data = pd.concat([plot_data, sector_data])
    
    if plot_data.empty:
        st.warning(f"⚠ Không có dữ liệu cho ngành '{selected_sector}' trong khoảng thời gian đã chọn.")
        return
    
    fig = go.Figure()
    for col in selected_columns:
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data[col],
            mode='lines',
            name=col
        ))
    
    fig.update_layout(
        title=f"Biểu đồ dữ liệu ngành '{selected_sector}'",
        xaxis=dict(title="Ngày"),
        yaxis=dict(title="Giá trị (tỷ VND)"),
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    st.plotly_chart(fig)

# Hàm tính SMA
def calculate_sma(data, window):
    return data['Price'].rolling(window=window, min_periods=1).mean()

# Hàm tính EMA
def calculate_ema(data, window):
    return data['Price'].ewm(span=window, adjust=False).mean()


# Hàm tính MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Price'].ewm(span=short_window, adjust=False, min_periods=1).mean()
    long_ema = data['Price'].ewm(span=long_window, adjust=False, min_periods=1).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Hàm tính RSI
def calculate_rsi(data, period=14):
    delta = data['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_stock_data_with_indicators(filtered_data, ticker, sma_window, ema_window):
    filtered_data = filtered_data.sort_values(by="Date").reset_index(drop=True)
    
    filtered_data["Index"] = filtered_data.index
    filtered_data["Month_Year"] = filtered_data["Date"].dt.to_period("M").dt.strftime("%m/%Y")

    unique_months = filtered_data["Month_Year"].unique()
    tickvals = []
    ticktext = []
    for month in unique_months:
        first_day_index = filtered_data[filtered_data["Month_Year"] == month].index[0]
        tickvals.append(first_day_index)
        ticktext.append(month)

    macd_line, signal_line, histogram = calculate_macd(filtered_data)
    sma_values = calculate_sma(filtered_data, sma_window)
    ema_values = calculate_ema(filtered_data, ema_window)
    rsi_values = calculate_rsi(filtered_data)

    # Tạo subplot với 3 hàng: Price, MACD, RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Biểu đồ giá
    fig.add_trace(go.Scatter(
        x=filtered_data["Index"],
        y=filtered_data["Price"],
        mode="lines",
        name="Price",
        line=dict(color="gold", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data["Index"],
        y=sma_values,
        mode="lines",
        name=f"SMA {sma_window}",
        line=dict(color="blue", width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data["Index"],
        y=ema_values,
        mode="lines",
        name=f"SMA {ema_window}",
        line=dict(color="green", width=1)
    ), row=1, col=1)

    colors = [
        'rgba(0,255,0,0.3)' if filtered_data["Volume"].iloc[i] > filtered_data["Volume"].iloc[i-1] 
        else 'rgba(255,0,0,0.3)' 
        for i in range(len(filtered_data))
    ]
    fig.add_trace(go.Bar(
        x=filtered_data["Index"],
        y=filtered_data["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.5
    ), row=1, col=1, secondary_y=True)

    # Biểu đồ MACD
    macd_colors = ['green' if val > 0 else 'red' for val in histogram]
    fig.add_trace(go.Bar(
        x=filtered_data["Index"],
        y=histogram,
        name="MACD Histogram",
        marker_color=macd_colors,
        opacity=0.5
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data["Index"],
        y=macd_line,
        mode="lines",
        name="MACD",
        line=dict(color="purple", width=1)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered_data["Index"],
        y=signal_line,
        mode="lines",
        name="Signal",
        line=dict(color="orange", width=1)
    ), row=2, col=1)


    fig.add_trace(go.Scatter(
        x=filtered_data["Index"],
        y=rsi_values,
        mode="lines",
        name="RSI",
        line=dict(color="cyan", width=1)
    ), row=3, col=1)


    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        title=f"📊 {ticker} - Price Analysis",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=120, b=50),
        height=800, 
        hovermode="x unified",
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            type="category",
            showgrid=False
        ),
        xaxis2=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            type="category",
            showgrid=False
        ),
        xaxis3=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            type="category",
            showgrid=False
        )
    )

    fig.update_yaxes(title_text="Price (VND)", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="MACD", row=2, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    return fig

import google.generativeai as genai

genai.configure(api_key="******") # hãy dùng API KEY của cậu nhé !!
model = genai.GenerativeModel("gemini-1.5-flash")

def get_ai_analysis(filtered_data, ticker, sma_window, ema_window):

    filtered_data = filtered_data.sort_values(by="Date").reset_index(drop=True)
    macd_line, signal_line, _ = calculate_macd(filtered_data)
    sma_values = calculate_sma(filtered_data, sma_window)
    ema_values = calculate_ema(filtered_data, ema_window)
    rsi_values = calculate_rsi(filtered_data)

    price_start = filtered_data['Price'].iloc[0]
    price_end = filtered_data['Price'].iloc[-1]
    price_trend = "tăng" if price_end > price_start else "giảm"

    last_10_days = filtered_data.tail(10)
    if len(last_10_days) < 10:
        return "Không đủ dữ liệu cho 10 ngày cuối."
    data_10_days = [
        f"{last_10_days['Date'].iloc[i].strftime('%Y-%m-%d')}: "
        f"Giá {last_10_days['Price'].iloc[i]:,.0f} VND, "
        f"Volume {last_10_days['Volume'].iloc[i]:,.0f}, "
        f"SMA {sma_values.tail(10).iloc[i]:,.0f}, "
        f"EMA {ema_values.tail(10).iloc[i]:,.0f}, "
        f"MACD {macd_line.tail(10).iloc[i]:,.2f}, "
        f"Signal {signal_line.tail(10).iloc[i]:,.2f}, "
        f"RSI {rsi_values.tail(10).iloc[i]:,.2f}"
        for i in range(10)
    ]

    # Prompt cho AI
    prompt = f"""
    Phân tích cổ phiếu {ticker} từ {filtered_data['Date'].iloc[0].strftime('%Y-%m-%d')} đến {filtered_data['Date'].iloc[-1].strftime('%Y-%m-%d')}:

    1. **Tổng quan giá**: Giá bắt đầu {price_start:,.0f} VND, kết thúc {price_end:,.0f} VND, xu hướng {price_trend}.

    2. **Phân tích 10 ngày cuối (đến {filtered_data['Date'].iloc[-1].strftime('%Y-%m-%d')})**:
       - **Dữ liệu**: {', '.join(data_10_days)}
       - **Chi tiết từng ngày**:
         - Giá: Ngày nào tăng/giảm so với ngày trước (ví dụ: "2025-03-20 tăng từ 24,000 lên 24,500 VND").
         - Price vs SMA {sma_window}/EMA {ema_window}: Ngày nào giá cắt lên/xuống SMA/EMA.
         - Price vs Volume: Khối lượng thay đổi thế nào khi giá tăng/giảm, có ngày nào đột biến (gấp 2 lần ngày trước).
         - MACD vs Signal: Ngày nào MACD cắt lên/xuống Signal, giá phản ứng ra sao.
         - RSI vs Price: Ngày nào RSI vào quá mua (>70) hoặc quá bán (<30), giá có đảo chiều không.

    Lưu ý: Phân tích ngắn gọn, rõ ràng, không đưa ra đề xuất đầu tư.
    """

    # Gọi Gemini API
    try:
        response = model.generate_content(prompt)
        return response.text if response and response.text else "Không nhận được phản hồi từ Gemini API."
    except Exception as e:
        return f"Lỗi khi gọi Gemini API: {str(e)}"


# Main function
def main():
    st.title("🌱 Stock Dashboard")
    data = load_data(FILE_PATH)
    ticker = st.sidebar.text_input("👉 Choose Code Stock")
    start_date = st.sidebar.date_input("📅 Start Date")
    end_date = st.sidebar.date_input("📅 End Date")

    st.sidebar.header("Technical Indicators Settings")
    sma_window = st.sidebar.selectbox("SMA Window", [10, 30, 50, 100], index=1)
    ema_window = st.sidebar.selectbox("EMA Window", [10, 30, 50, 100], index=1)

    if ticker:
        filtered_data = filter_data(data, ticker, pd.to_datetime(start_date), pd.to_datetime(end_date))
        if not filtered_data.empty:
            st.plotly_chart(plot_stock_data_with_indicators(filtered_data, ticker, sma_window, ema_window))
            st.write(filtered_data)
            
            # Thêm phần nhận xét từ AI
            st.subheader("🤖 Nhận xét từ AI")
            try:
                ai_analysis = get_ai_analysis(filtered_data, ticker, sma_window, ema_window)
                st.write(ai_analysis)
            except Exception as e:
                st.error(f"Không thể tạo nhận xét từ AI: {str(e)}")
        else:
            st.warning("⚠ No data found for the selected stock and date range.")
    else:
        st.info("ℹ Please enter a stock code to view the data.")
    
    fundamental_tab = st.tabs(["Fundamental Data"])
    with fundamental_tab[0]:
        if ticker:
            display_fundamental_data(ticker, pd.to_datetime(start_date), pd.to_datetime(end_date))
        else:
            st.info("ℹ Please enter a stock code to view fundamental data.")

if __name__ == "__main__":
    main()

