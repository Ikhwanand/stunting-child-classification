import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

import joblib

import warnings
from agent import stunting_recommendation

warnings.filterwarnings("ignore")

# Configuration
st.set_page_config(
    page_title="Klasifikasi Status Gizi Balita",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Fungsi untuk memuat model dan encoder
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("./models/rf_models.joblib")
        le_gender = joblib.load("./models/le_gender.pkl")
        le_status = joblib.load("./models/le_status.pkl")
        scaler = joblib.load("./models/scaler.pkl")
        return rf_model, le_gender, le_status, scaler
    except:
        return None, None, None, None


# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("./data/data_balita.csv")
        return df
    except:
        # Data dummy jika file tidak ada
        np.random.seed(42)
        n_samples = 1000
        data = {
            "Umur (bulan)": np.random.randint(0, 60, n_samples),
            "Jenis Kelamin": np.random.choice(["laki-laki", "perempuan"], n_samples),
            "Tinggi Badan (cm)": np.random.normal(85, 15, n_samples),
            "Status Gizi": np.random.choice(
                ["normal", "stunted", "severly stunted", "tinggi"], n_samples
            ),
        }
        return pd.DataFrame(data)


# Header utama
st.markdown(
    "<h1 class='main-header'>👶 Aplikasi Klasifikasi Status Gizi Balita</h1>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🔧 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    [
        "🏠 Dashboard",
        "📊 Analisis Data",
        "🤖 Prediksi",
        "📈 Visualisasi",
        "ℹ️ Info Model",
    ],
)

# Load data dan model
df = load_data()
rf_model, le_gender, le_status, scaler = load_models()


# Dashboard
if menu == "🏠 Dashboard":
    st.markdown(
        "<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Data", value=f"{len(df):,}", delta="Data Balita")

    with col2:
        normal_count = len(df[df["Status Gizi"] == "normal"])
        st.metric(
            label="Status Normal",
            value=f"{normal_count:,}",
            delta=f"{normal_count/len(df)*100:.1f}%",
        )

    with col3:
        stunted_count = len(df[df["Status Gizi"].isin(["stunted", "severly stunted"])])
        st.metric(
            label="Stunting",
            value=f"{stunted_count:,}",
            delta=f"{stunted_count/len(df)*100:.1f}%",
        )

    with col4:
        avg_age = df["Umur (bulan)"].mean()
        st.metric(label="Rata-rata Umur", value=f"{avg_age:.1f}", delta="bulan")

    # Grafik distribusi status gizi
    st.markdown(
        "<h3 class='sub-header'>Distribusi Status Gizi</h3>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            df,
            names="Status Gizi",
            title="Distribusi Status Gizi Balita",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        status_counts = df["Status Gizi"].value_counts()
        fig_bar = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Jumlah Balita per Status Gizi",
            labels={"x": "Status Gizi", "y": "Jumlah"},
            color=status_counts.values,
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# Analisis data
elif menu == "📊 Analisis Data":
    st.markdown(
        "<h2 class='sub-header'>Analisis Eksploratori Data</h2>", unsafe_allow_html=True
    )

    # Informasi dataset
    st.markdown("### 📋 Informasi Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dimensi Dataset:**")
        st.write(f"- Jumlah baris: {df.shape[0]:,}")
        st.write(f"- Jumlah kolom: {df.shape[1]}")

        st.write("**Tipe Data:**")
        for col, dtype in df.dtypes.items():
            st.write(f"- {col}: {dtype}")

    with col2:
        st.write("**Statistik Deskriptif:**")
        st.dataframe(df.describe())

    # Korelasi
    st.markdown("### 🔗 Matriks Korelasi")

    # Encode categorical variables untuk korelasi
    df_encoded = df.copy()
    df_encoded["Jenis Kelamin"] = le_gender.transform(df["Jenis Kelamin"])
    df_encoded["Status Gizi"] = le_status.transform(df["Status Gizi"])

    corr_matrix = df_encoded.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matriks Korelasi Antar Variabel",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Distribusi berdasarkan jenis kelamin
    st.markdown("### 👫 Analisis Berdasarkan Jenis Kelamin")

    col1, col2 = st.columns(2)

    with col1:
        fig_gender = px.histogram(
            df,
            x="Status Gizi",
            color="Jenis Kelamin",
            title="Distribusi Status Gizi berdasarkan Jenis Kelamin",
            barmode="group",
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        fig_box = px.box(
            df,
            x="Status Gizi",
            y="Tinggi Badan (cm)",
            color="Jenis Kelamin",
            title="Distribusi Tinggi Badan berdasarkan Status Gizi",
        )
        st.plotly_chart(fig_box, use_container_width=True)


# Prediksi
elif menu == "🤖 Prediksi":
    st.markdown(
        "<h2 class='sub-header'>Prediksi Status Gizi Balita</h2>",
        unsafe_allow_html=True,
    )

    # st.success("✅ Model siap digunakan!")

    # Form input prediksi
    st.markdown("### 📝 Input Data Balita")

    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input(
            "Umur (bulan)",
            min_value=0,
            max_value=60,
            value=24,
            help="Masukkan umur balita dalam bulan (0-60)",
        )

        jenis_kelamin = st.selectbox(
            "Jenis Kelamin",
            options=["laki-laki", "perempuan"],
            help="Pilih jenis kelamin balita",
        )

    with col2:
        tinggi_badan = st.number_input(
            "Tinggi Badan (cm)",
            min_value=40.0,
            max_value=120.0,
            value=85.0,
            step=0.1,
            help="Masukkan tinggi badan balita dalam cm",
        )

        negara = st.text_input(
            "Negara", value="Indonesia", help="Masukkan negara tempat tinggal balita"
        )

    if st.button("🔮 Prediksi Status Gizi", type="primary"):
        try:
            # Prepare input data
            jenis_kelamin_encoded = le_gender.transform([jenis_kelamin])[0]
            input_data = np.array([[umur, jenis_kelamin_encoded, tinggi_badan]])
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = rf_model.predict(input_data_scaled)[0]
            prediction_proba = rf_model.predict_proba(input_data_scaled)[0]

            # Decode prediction
            status_predicted = le_status.inverse_transform([prediction])[0]

            # Display results
            st.markdown("### 🎯 Hasil Prediksi")

            # Status prediction with color coding
            status_colors = {
                "normal": "green",
                "stunted": "orange",
                "severly stunted": "red",
                "tinggi": "blue",
            }

            color = status_colors.get(status_predicted, "gray")
            st.markdown(
                f'**Status Gizi Prediksi:** <span style="color:{color}; font-size: 1.2em; font-weight:bold">{status_predicted.upper()}</span>',
                unsafe_allow_html=True,
            )

            # Probability distribution
            st.markdown("### 📊 Tingkat Kepercayaan Prediksi")

            prob_df = pd.DataFrame(
                {"Status": le_status.classes_, "Probabilitas": prediction_proba}
            ).sort_values("Probabilitas", ascending=False)

            fig_prob = px.bar(
                prob_df,
                x="Status",
                y="Probabilitas",
                title="Distribusi Probabilitas Prediksi",
                color="Probabilitas",
                color_continuous_scale="viridis",
            )
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

            # Recommendation
            st.markdown("### 💡 Rekomendasi")
            if status_predicted == "severely stunted":
                st.error(
                    "⚠️ **Perhatikan Khusus Diperlukan!** Balita mengalami stunting berat. Segera konsultasi dengan tenaga kesehatan."
                )

                kelamin = "boy" if jenis_kelamin == "laki-laki" else "girl"

                with st.spinner("Mohon tunggu..."):
                    recommendation = stunting_recommendation(
                        age=umur,
                        height_cm=tinggi_badan,
                        gender=kelamin,
                        country=negara,
                        disease=status_predicted,
                    )

                    # st.markdown('## 📝 Rekomendasi')
                    st.markdown(recommendation)
            elif status_predicted == "stunted":
                st.warning(
                    "⚠️ **Perlu Perhatian!** Balita mengalami stunting. Perbaiki asupan gizi dan konsultasi dengan tenaga kesehatan."
                )

                kelamin = "boy" if jenis_kelamin == "laki-laki" else "girl"

                with st.spinner("Mohon tunggu..."):
                    recommendation = stunting_recommendation(
                        age=umur,
                        height_cm=tinggi_badan,
                        gender=kelamin,
                        country=negara,
                        disease=status_predicted,
                    )

                    # st.markdown('## 📝 Rekomendasi')
                    st.markdown(recommendation)
            elif status_predicted == "normal":
                st.success(
                    "✅ **Status Normal!** Pertahankan pola makan dan gaya hidup sehat."
                )
            else:
                st.info(
                    "ℹ️ **Status Tinggi!** Balita memiliki tinggi badan di atas rata-rata. Tetap jaga keseimbangan gizi."
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam prediksi: {str(e)}")


# Visualisasi
elif menu == "📈 Visualisasi":
    st.markdown(
        "<h2 class='sub-header'>Visualisasi Data Interaktif</h2>",
        unsafe_allow_html=True,
    )

    # Pilihan visualisasi
    viz_type = st.selectbox(
        "Pilih Jenis Visualisasi:", ["Scatter Plot", "Box Plot", "Histogram"]
    )

    if viz_type == "Scatter Plot":

        fig_scatter = px.scatter(
            df,
            x="Umur (bulan)",
            y="Tinggi Badan (cm)",
            color="Status Gizi",
            hover_data=["Jenis Kelamin"],
            title="Scatter Plot: Umur (bulan) vs Tinggi Badan (cm)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif viz_type == "Box Plot":
        variable = st.selectbox(
            "Pilih Variabel:", ["Umur (bulan)", "Tinggi Badan (cm)"]
        )

        fig_box = px.box(
            df,
            x="Status Gizi",
            y=variable,
            color="Jenis Kelamin",
            title=f"Box Plot: {variable} berdasarkan Status Gizi",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    elif viz_type == "Histogram":
        variable = st.selectbox(
            "Pilih Variabel:", ["Umur (bulan)", "Tinggi Badan (cm)"]
        )

        fig_hist = px.histogram(
            df,
            x=variable,
            color="Status Gizi",
            marginal="box",
            title=f"Distribusi {variable}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)


elif menu == "ℹ️ Info Model":
    st.markdown(
        '<h2 class="sub-header">Informasi Model Machine Learning</h2>',
        unsafe_allow_html=True,
    )

    if rf_model is not None:
        st.success("✅ Model Random Forest berhasil dimuat!")

        # Model parameters
        st.markdown("### 🔧 Parameter Model")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Algoritma:** Random Forest")
            st.write(f"**Jumlah Trees:** {rf_model.n_estimators}")
            st.write(f"**Max Depth:** {rf_model.max_depth}")
            st.write(f"**Min Samples Split:** {rf_model.min_samples_split}")

        with col2:
            st.write(f"**Min Samples Leaf:** {rf_model.min_samples_leaf}")
            st.write(f"**Random State:** {rf_model.random_state}")
            st.write(f"**Bootstrap:** {rf_model.bootstrap}")

        # Feature importance
        if hasattr(rf_model, "feature_importances_"):
            st.markdown("### 📊 Feature Importance")

            feature_names = ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"]
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": rf_model.feature_importances_}
            ).sort_values("Importance", ascending=False)

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance dalam Model",
                color="Importance",
                color_continuous_scale="viridis",
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            # Feature importance table
            st.dataframe(importance_df, use_container_width=True)

        # Model performance (jika ada data test)
        st.markdown("### 📈 Informasi Tambahan")
        st.info(
            """
        **Tentang Model Random Forest:**
        - Random Forest adalah ensemble method yang menggabungkan multiple decision trees
        - Robust terhadap overfitting dan dapat menangani missing values
        - Memberikan feature importance yang berguna untuk interpretasi
        - Cocok untuk klasifikasi multi-class seperti status gizi balita
        """
        )

    else:
        st.warning(
            "⚠️ Model belum tersedia. Silahkan latih model terlebih dahulu di menu Prediksi."
        )


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏥 Aplikasi Klasifikasi Status Gizi Balita | Dibuat dengan ❤️ menggunakan Streamlit</p>
        <p><small>⚠️ Disclaimer: Aplikasi ini hanya untuk tujuan edukasi dan tidak menggantikan konsultasi medis profesional</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
