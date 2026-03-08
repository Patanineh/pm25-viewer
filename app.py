import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import datetime

# 1. ตั้งค่าหน้าเพจแบบเต็มจอ
st.set_page_config(page_title="PM2.5 Prediction Viewer | ปังปอนด์999", page_icon="☁️", layout="wide")

# ซ่อนแถบด้านบน (ปุ่ม Deploy และจุด 3 จุด)
st.markdown("""
<style>
    header {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# แถบควบคุมด้านซ้ายมือ (Sidebar)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=100)

# สร้างสวิตช์สลับโหมด สว่าง-มืด (ค่าเริ่มต้น value=False คือโหมดสว่าง)
st.sidebar.markdown("")
dark_mode = st.sidebar.toggle("🌙 โหมดมืด (Dark Mode)", value=False)

# อัปเกรดระบบเปลี่ยนสี (Dark/Light Mode)
if dark_mode:
    template_theme = "plotly_dark"
    st.markdown("""
    <style>
        .stApp {background-color: #0E1117; color: #FAFAFA;}
        [data-testid="stSidebar"] {background-color: #262730;}
        h1, h2, h3, h4, h5, h6 {color: #FAFAFA !important;}
        [data-testid="stMetricValue"] div {color: #FAFAFA !important;}
        [data-testid="stMetricLabel"] p {color: #A1A1A1 !important; font-size: 16px !important;}
        .stMarkdown p {color: #E0E0E0;}
        div[data-baseweb="calendar"] {background-color: #262730;}
    </style>
    """, unsafe_allow_html=True)
else:
    template_theme = "plotly_white"
    st.markdown("""
    <style>
        .stApp {background-color: #FFFFFF; color: #31333F;}
        [data-testid="stSidebar"] {background-color: #F0F2F6;}
        h1, h2, h3, h4, h5, h6 {color: #31333F !important;}
        [data-testid="stMetricValue"] div {color: #31333F !important;}
        [data-testid="stMetricLabel"] p {color: #6C757D !important; font-size: 16px !important;}
        .stMarkdown p {color: #31333F;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# พื้นที่หลักของโปรแกรม
# ==========================================
st.title("☁️ PM2.5 Prediction Viewer")
st.markdown("**โดย กลุ่ม ปังปอนด์999**")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('air_quality_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

st.sidebar.header("📂 Data Upload (Unseen Data)")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV เพื่อประมวลผล", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # --- Data Preparation ---
    df['convert time'] = pd.to_datetime(df['convert time'], errors='coerce')
    
    numeric_cols = ["pm1", "pm2.5", "pm4", "pm10", "wind", "rain drop"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=numeric_cols)
    
    if len(df) > 0:
        features = ["pm1", "pm2.5", "pm4", "pm10", "wind", "rain drop"]
        X_unseen = df[features]
        
        if model:
            # ทำนายค่า PM2.5
            df['Predicted_PM2.5'] = model.predict(X_unseen)
            
            # --- ดักจับค่าเพี้ยน (Clipping) ---
            df['Predicted_PM2.5'] = df['Predicted_PM2.5'].clip(lower=0, upper=150)
            
            min_date = df['convert time'].min().date()
            max_date = df['convert time'].max().date()
            
            st.markdown("---")
            
            row1_col1, row1_col2 = st.columns([1, 3]) 
            
            with row1_col1:
                selected_date = st.date_input(
                    "เลือกวัน (แสดงย้อนหลัง 7 วัน)", 
                    value=max_date,
                    min_value=datetime.date(2020, 1, 1),
                    max_value=datetime.date(2030, 12, 31)
                )
                
            start_date = selected_date - datetime.timedelta(days=6)
                
            with row1_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:gray;'>ช่วงข้อมูลทั้งหมดที่มีในไฟล์: {min_date} ถึง {max_date}</span>", unsafe_allow_html=True)

            filtered_df = df[(df['convert time'].dt.date >= start_date) & (df['convert time'].dt.date <= selected_date)]
            
            st.markdown("<br>", unsafe_allow_html=True) 
            
            if not filtered_df.empty:
                # ==========================================
                # ส่วนแสดงสถิติ PM2.5
                # ==========================================
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                
                avg_pm = filtered_df['Predicted_PM2.5'].mean()
                min_pm = filtered_df['Predicted_PM2.5'].min()
                max_pm = filtered_df['Predicted_PM2.5'].max()
                actual_days = len(filtered_df['convert time'].dt.date.unique())
                
                row2_col1.metric(label=f"PM2.5 เฉลี่ย ({actual_days} วัน)", value=f"{avg_pm:.2f}")
                row2_col2.metric(label="PM2.5 ต่ำสุด", value=f"{min_pm:.2f}")
                row2_col3.metric(label="PM2.5 สูงสุด", value=f"{max_pm:.2f}")
                
                # ==========================================
                # ส่วนแสดงสถิติ ลม และ ฝน
                # ==========================================
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<span style='color:gray; font-size: 0.9em;'>สภาพแวดล้อม (ลมและฝน) ในช่วงเวลาที่เลือก</span>", unsafe_allow_html=True)
                row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
                
                avg_wind = filtered_df['wind'].mean()
                max_wind = filtered_df['wind'].max()
                avg_rain = filtered_df['rain drop'].mean()
                max_rain = filtered_df['rain drop'].max()
                
                row3_col1.metric(label="ความเร็วลมเฉลี่ย", value=f"{avg_wind:.2f} m/s")
                row3_col2.metric(label="ความเร็วลมสูงสุด", value=f"{max_wind:.2f} m/s")
                row3_col3.metric(label="ปริมาณฝนเฉลี่ย", value=f"{avg_rain:.2f} mm")
                row3_col4.metric(label="ปริมาณฝนสูงสุด", value=f"{max_rain:.2f} mm")
                
                st.markdown("<br>", unsafe_allow_html=True) 

                # ==========================================
                # วาดกราฟ (กลับมาเป็นเส้นเรียบง่าย แต่ลดการสวิง)
                # ==========================================
                graph_df = filtered_df.copy()
                graph_df = graph_df.set_index('convert time').resample('H')[['Predicted_PM2.5']].mean().reset_index()
                graph_df = graph_df.dropna()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=graph_df['convert time'], 
                    y=graph_df['Predicted_PM2.5'], 
                    mode='lines', 
                    name='PM2.5 (เฉลี่ยรายชั่วโมง)',  
                    line=dict(color='#0077b6', width=2) # กลับมาใช้เส้นปกติ สีฟ้าเข้ม
                ))
                
                fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Danger (>50)")
                fig.add_hline(y=37.5, line_dash="dash", line_color="orange", annotation_text="Warning (>37.5)")
                
                fig.update_layout(
                    title=f"แนวโน้ม PM2.5 ตั้งแต่วันที่ {start_date} ถึง {selected_date}",
                    yaxis_title="ปริมาณ PM2.5 (µg/m³)",
                    xaxis_title="วันที่",              
                    template=template_theme,
                    margin=dict(l=0, r=0, t=40, b=0),
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                if not dark_mode:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                else:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#444')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#444')

                st.plotly_chart(fig, use_container_width=True)
                
                # ==========================================
                # ส่วนแสดงตารางข้อมูล
                # ==========================================
                st.markdown("**📊 ข้อมูลรายละเอียดประจำวัน**")
                
                display_df = filtered_df.copy()
                display_df['วันที่'] = display_df['convert time'].dt.date
                daily_df = display_df.groupby('วันที่')[['pm1', 'pm2.5', 'pm4', 'pm10', 'wind', 'rain drop', 'Predicted_PM2.5']].mean().reset_index()
                daily_df = daily_df.round(2)
                
                daily_df.rename(columns={
                    'pm1': 'PM1',
                    'pm2.5': 'PM2.5 จริง',
                    'pm4': 'PM4',
                    'pm10': 'PM10',
                    'wind': 'ลม (m/s)',
                    'rain drop': 'ฝน (mm)',
                    'Predicted_PM2.5': 'PM2.5 ทำนาย'
                }, inplace=True)
                
                st.dataframe(daily_df, use_container_width=True, hide_index=True)
                
            else:
                st.warning(f"⚠️ ไม่มีข้อมูลในช่วงวันที่ {start_date} ถึง {selected_date} กรุณาเปลี่ยนวันที่ในปฏิทิน")
            
            # --- แผนที่ ---
            st.markdown("---")
            st.subheader("📍 ตำแหน่งสถานีตรวจวัด (มจพ. ปราจีนบุรี)")
            m = folium.Map(location=[14.159013, 101.346016], zoom_start=15)
            folium.Marker(
                [14.159013, 101.346016],
                popup="สถานีตรวจวัด IoT: มจพ. ปราจีนบุรี",
                icon=folium.Icon(color="red", icon="cloud")
            ).add_to(m)
            st_folium(m, width=1200, height=400)
            
        else:
            st.error("⚠️ ไม่พบไฟล์โมเดล `air_quality_model.pkl` ในโฟลเดอร์เดียวกัน")
            
    else:
        st.error("ไม่พบข้อมูลที่ใช้ในการทำนายครบถ้วนในไฟล์ที่อัปโหลด")
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้ายมือ")