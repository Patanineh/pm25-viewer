import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import datetime

# 1. ตั้งค่าหน้าเพจแบบเต็มจอ
st.set_page_config(page_title="ระบบพยากรณ์ PM2.5 ล่วงหน้า 7 วัน | ปังปอนด์999", page_icon="🔮", layout="wide")

# ซ่อนแถบเมนูเดิม
st.markdown("<style>header {visibility: hidden !important;}</style>", unsafe_allow_html=True)

# ==========================================
# แถบควบคุมด้านซ้ายมือ (Sidebar)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=80)
st.sidebar.title("🔮 ศูนย์พยากรณ์อากาศ")

# เลือกโหมดการแสดงผล
dark_mode = st.sidebar.toggle("🌙 โหมดกลางคืน (Dark Mode)", value=False)

if dark_mode:
    template_theme = "plotly_dark"
    st.markdown("""<style>
        .stApp {background-color: #0E1117; color: #FAFAFA;}
        [data-testid="stSidebar"] {background-color: #262730;}
        h1, h2, h3, h4, h5, h6 {color: #FAFAFA !important;}
        [data-testid="stMetricValue"] div {color: #00B4D8 !important;}
        .stMarkdown p {color: #E0E0E0;}
    </style>""", unsafe_allow_html=True)
else:
    template_theme = "plotly_white"
    st.markdown("""<style>
        .stApp {background-color: #FFFFFF; color: #31333F;}
        [data-testid="stSidebar"] {background-color: #F0F2F6;}
    </style>""", unsafe_allow_html=True)

st.title("🔮 ระบบพยากรณ์ PM2.5 ล่วงหน้า 7 วัน")
st.markdown("**โดย กลุ่ม ปังปอนด์999** | *พยากรณ์ค่าฝุ่นละอองจากปัจจัยสภาพอากาศ*")

@st.cache_resource
def load_model():
    try:
        # โหลดโมเดล RidgeCV Pipeline
        return joblib.load('air_quality_model.pkl')
    except:
        return None

model = load_model()

st.sidebar.header("📂 ข้อมูลสำหรับประมวลผล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV (Data สำหรับพยากรณ์)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['convert time'] = pd.to_datetime(df['convert time'], errors='coerce')
    
    # ตัวแปรที่ใช้ (ต้องตรงกับโมเดล)
    features = ["pm1", "pm2.5", "pm4", "pm10", "wind", "rain drop"]
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=features)
    
    if len(df) > 0 and model:
        # คำนวณพยากรณ์
        df['Forecast_PM2.5'] = model.predict(df[features])
        df['Forecast_PM2.5'] = df['Forecast_PM2.5'].clip(lower=0, upper=150)
        
        min_data_date = df['convert time'].min().date()
        max_data_date = df['convert time'].max().date()
        
        st.markdown("---")
        
        # ส่วนเลือกวันที่เริ่มพยากรณ์
        col_date, col_info = st.columns([1, 2])
        with col_date:
            # ตั้งค่าเริ่มต้นเป็นวันที่เร็วที่สุดในไฟล์ เพื่อให้เห็นข้อมูล 7 วันข้างหน้า
            selected_date = st.date_input("🗓️ วันที่เริ่มต้นพยากรณ์", value=min_data_date)
        
        # คำนวณวันสิ้นสุด (Selected Date + 6 วัน = รวมเป็น 7 วัน)
        end_date = selected_date + datetime.timedelta(days=6)
        
        filtered_df = df[(df['convert time'].dt.date >= selected_date) & (df['convert time'].dt.date <= end_date)]
        
        if not filtered_df.empty:
            # ==========================================
            # สรุปภาพรวมการพยากรณ์ 7 วัน
            # ==========================================
            st.subheader(f"📊 ผลการพยากรณ์ล่วงหน้า: {selected_date} ถึง {end_date}")
            
            m1, m2, m3 = st.columns(3)
            avg_forecast = filtered_df['Forecast_PM2.5'].mean()
            max_forecast = filtered_df['Forecast_PM2.5'].max()
            
            m1.metric("ค่าพยากรณ์เฉลี่ย 7 วัน", f"{avg_forecast:.2f} µg/m³")
            m2.metric("ค่าพยากรณ์สูงสุดที่คาดว่าจะเกิด", f"{max_forecast:.2f} µg/m³")
            m3.metric("ความเร็วลมเฉลี่ย", f"{filtered_df['wind'].mean():.2f} m/s")

            # ==========================================
            # กราฟพยากรณ์ล่วงหน้า 7 วัน
            # ==========================================
            graph_df = filtered_df.set_index('convert time').resample('H')[['Forecast_PM2.5']].mean().reset_index().dropna()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=graph_df['convert time'], 
                y=graph_df['Forecast_PM2.5'], 
                mode='lines',
                name='พยากรณ์ PM2.5',
                line=dict(color='#00B4D8', width=3)
            ))
            
            # เส้นเกณฑ์มาตรฐาน
            fig.add_hline(y=37.5, line_dash="dot", line_color="#F77F00", annotation_text="เริ่มมีผลต่อสุขภาพ")
            fig.add_hline(y=50, line_dash="dot", line_color="#D62828", annotation_text="อันตราย")
            
            fig.update_layout(
                title=f"📈 แนวโน้มพยากรณ์ PM2.5 ล่วงหน้า 7 วัน (รายชั่วโมง)",
                yaxis_title="PM2.5 (µg/m³)",
                template=template_theme,
                height=400,
                margin=dict(l=0, r=0, t=50, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ==========================================
            # ตารางพยากรณ์รายวัน
            # ==========================================
            st.markdown("**📝 ตารางพยากรณ์รายวัน (7 วันข้างหน้า)**")
            daily = filtered_df.copy()
            daily['วันที่'] = daily['convert time'].dt.date
            daily_summary = daily.groupby('วันที่')[['wind', 'rain drop', 'Forecast_PM2.5']].mean().reset_index().round(2)
            daily_summary.columns = ['วันที่พยากรณ์', 'ความเร็วลมคาดการณ์ (m/s)', 'ปริมาณฝนคาดการณ์ (mm)', 'พยากรณ์ PM2.5']
            
            st.dataframe(daily_summary, use_container_width=True, hide_index=True)
            
            # ==========================================
            # แผนที่
            # ==========================================
            st.markdown("---")
            st.subheader("📍 สถานีตรวจวัดและพยากรณ์")
            m = folium.Map(location=[14.159013, 101.346016], zoom_start=15)
            folium.Marker(
                [14.159013, 101.346016], 
                popup="มจพ. ปราจีนบุรี",
                icon=folium.Icon(color="blue", icon="cloud")
            ).add_to(m)
            st_folium(m, width=1200, height=300)
            
        else:
            st.warning(f"⚠️ ไม่พบข้อมูลพยากรณ์ตั้งแต่ชุดวันที่ {selected_date} เป็นต้นไป (ข้อมูลในไฟล์สิ้นสุดที่ {max_data_date})")
    else:
        st.error("⚠️ ไม่สามารถพยากรณ์ได้: กรุณาตรวจสอบชื่อคอลัมน์ใน CSV หรือไฟล์โมเดล")
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ข้อมูลเพื่อเริ่มระบบพยากรณ์ล่วงหน้า 7 วัน")
