import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import datetime

# 1. ตั้งค่าหน้าเพจแบบเต็มจอ
st.set_page_config(page_title="ระบบพยากรณ์คุณภาพอากาศ | ปังปอนด์999", page_icon="🔮", layout="wide")

# ซ่อนแถบเมนูเดิม
st.markdown("""
<style>
    header {visibility: hidden !important;}
    .main {background-color: transparent;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# แถบควบคุมด้านซ้ายมือ (Sidebar)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=80)
st.sidebar.title("🔮 เมนูหลัก")

# สลับโหมด สว่าง-มืด
dark_mode = st.sidebar.toggle("🌙 โหมดมืด (Dark Mode)", value=False)

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
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# ฟังก์ชันโหลดโมเดล
# ==========================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('air_quality_model.pkl')
    except:
        return None

model = load_model()

# ==========================================
# พื้นที่หลักของโปรแกรม
# ==========================================
st.title("🔮 ระบบวิเคราะห์และพยากรณ์คุณภาพอากาศ")
st.markdown("**โดย กลุ่ม ปังปอนด์999**")

st.sidebar.header("📂 จัดการข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV สำหรับพยากรณ์", type=['csv'])

if uploaded_file is not None:
    # 1. แสดงข้อมูลดิบ (Raw Data)
    df_raw = pd.read_csv(uploaded_file)
    with st.expander("📄 เปิดดูข้อมูลดิบในไฟล์ CSV (Raw Data)"):
        st.dataframe(df_raw, use_container_width=True)

    # 2. เตรียมข้อมูลประมวลผล
    df = df_raw.copy()
    df['convert time'] = pd.to_datetime(df['convert time'], errors='coerce')
    
    # ตัวแปรที่ต้องใช้ในโมเดล
    features = ["pm1", "pm2.5", "pm4", "pm10", "wind", "rain drop"]
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=features)
    
    if len(df) > 0 and model:
        # ทำการพยากรณ์
        df['Forecast_PM2.5'] = model.predict(df[features])
        df['Forecast_PM2.5'] = df['Forecast_PM2.5'].clip(lower=0, upper=150)
        
        # ค้นหาวันสุดท้ายที่มีค่าจริง (pm2.5) ในไฟล์
        last_actual_date = df[df['pm2.5'].notna()]['convert time'].max().date()
        
        st.markdown("---")
        
        # ส่วนเลือกวันที่ (พยากรณ์ล่วงหน้า 7 วัน)
        col_date, col_status = st.columns([1, 2])
        with col_date:
            selected_date = st.date_input("🗓️ วันที่เริ่มต้นวิเคราะห์", value=df['convert time'].min().date())
        
        end_date = selected_date + datetime.timedelta(days=6)
        mask = (df['convert time'].dt.date >= selected_date) & (df['convert time'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        
        if not filtered_df.empty:
            # ------------------------------------------
            # ส่วนที่ 1: สถิติ PM2.5 (พยากรณ์)
            # ------------------------------------------
            r2 = st.columns(3)
            actual_days_count = len(filtered_df['convert time'].dt.date.unique())
            r2[0].metric(label=f"PM2.5 พยากรณ์เฉลี่ย ({actual_days_count} วัน)", value=f"{filtered_df['Forecast_PM2.5'].mean():.2f}")
            r2[1].metric(label="PM2.5 พยากรณ์ต่ำสุด", value=f"{filtered_df['Forecast_PM2.5'].min():.2f}")
            r2[2].metric(label="PM2.5 พยากรณ์สูงสุด", value=f"{filtered_df['Forecast_PM2.5'].max():.2f}")
            
            # ------------------------------------------
            # ส่วนที่ 2: สถิติสภาพแวดล้อม (ตัวเล็ก)
            # ------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<span style='color:gray; font-size: 0.9em;'>🍃 สภาพแวดล้อม (ลมและฝน) ในช่วงเวลาที่เลือก</span>", unsafe_allow_html=True)
            r3 = st.columns(4)
            r3[0].metric(label="ความเร็วลมเฉลี่ย", value=f"{filtered_df['wind'].mean():.2f} m/s")
            r3[1].metric(label="ความเร็วลมสูงสุด", value=f"{filtered_df['wind'].max():.2f} m/s")
            r3[2].metric(label="ปริมาณฝนเฉลี่ย", value=f"{filtered_df['rain drop'].mean():.2f} mm")
            r3[3].metric(label="ปริมาณฝนสูงสุด", value=f"{filtered_df['rain drop'].max():.2f} mm")
            
            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------
            # ส่วนที่ 3: กราฟแยกวิเคราะห์
            # ------------------------------------------
            # กราฟพยากรณ์
            st.subheader("🔮 1. กราฟพยากรณ์ PM2.5 (Model Forecast)")
            graph_forecast = filtered_df.set_index('convert time').resample('H')[['Forecast_PM2.5']].mean().reset_index().dropna()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=graph_forecast['convert time'], y=graph_forecast['Forecast_PM2.5'], 
                                     mode='lines', name='ค่าพยากรณ์', line=dict(color='#00B4D8', width=3)))
            fig1.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Danger (>50)")
            fig1.add_hline(y=37.5, line_dash="dash", line_color="orange", annotation_text="Warning (>37.5)")
            fig1.update_layout(yaxis_title="PM2.5 (µg/m³)", xaxis_title="วันที่", template=template_theme, height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)

            # กราฟค่าจริง
            st.subheader("📡 2. กราฟค่าจริงจากเซ็นเซอร์ (Actual Data)")
            if selected_date > last_actual_date:
                st.info(f"💡 **ไม่มีค่าจริงมาเปรียบเทียบ:** เนื่องจากในไฟล์ข้อมูลจริงสิ้นสุดวันที่ {last_actual_date}")
            else:
                graph_actual = filtered_df.set_index('convert time').resample('H')[['pm2.5']].mean().reset_index().dropna()
                if not graph_actual.empty:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=graph_actual['convert time'], y=graph_actual['pm2.5'], 
                                             mode='lines', name='ค่าจริง', line=dict(color='#6c757d', width=2)))
                    fig2.update_layout(yaxis_title="PM2.5 (µg/m³)", xaxis_title="วันที่", template=template_theme, height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("⚠️ ไม่พบข้อมูลจริงในช่วงวันที่เลือก")

            # ------------------------------------------
            # ส่วนที่ 4: ตารางสรุปรายวัน 7 วัน
            # ------------------------------------------
            st.markdown("---")
            st.markdown("**📊 ข้อมูลรายละเอียดประจำวัน (สรุปรายวัน)**")
            daily = filtered_df.copy()
            daily['วันที่'] = daily['convert time'].dt.date
            daily_summary = daily.groupby('วันที่')[['pm1', 'pm2.5', 'pm4', 'pm10', 'wind', 'rain drop', 'Forecast_PM2.5']].mean().reset_index().round(2)
            
            daily_summary.rename(columns={
                'วันที่': 'วันที่พยากรณ์',
                'pm1': 'PM1',
                'pm2.5': 'PM2.5 จริง',
                'pm4': 'PM4',
                'pm10': 'PM10',
                'wind': 'ลม (m/s)',
                'rain drop': 'ฝน (mm)',
                'Forecast_PM2.5': 'PM2.5 พยากรณ์'
            }, inplace=True)
            
            st.dataframe(daily_summary, use_container_width=True, hide_index=True)
            
            # ------------------------------------------
            # ส่วนที่ 5: แผนที่
            # ------------------------------------------
            st.markdown("---")
            st.subheader("📍 ตำแหน่งสถานีตรวจวัด (มจพ. ปราจีนบุรี)")
            m = folium.Map(location=[14.159013, 101.346016], zoom_start=15)
            folium.Marker([14.159013, 101.346016], popup="สถานี: มจพ. ปราจีนบุรี", icon=folium.Icon(color="red", icon="cloud")).add_to(m)
            st_folium(m, width=1200, height=300)
            
        else:
            st.warning(f"⚠️ ไม่พบข้อมูลในช่วงวันที่เลือก (ไฟล์มีข้อมูลถึง {df['convert time'].max().date()})")
    else:
        st.error("⚠️ ข้อมูลไม่ครบถ้วนหรือไฟล์โมเดลมีปัญหา")
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ CSV เพื่อเข้าสู่ระบบวิเคราะห์และพยากรณ์")
