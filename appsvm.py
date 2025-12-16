import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import base64
from pathlib import Path

# --------------------------
# 1. GLOBAL STYLING & CONFIG
# --------------------------

# Set Page Config
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa (SVM)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
GREEN_COLOR = "#28A745"
DARK_GREEN = "#1E7E34"
GRAY_LIGHT = "#f8f9fa"

# List Anggota Kelompok (Asumsi file foto ada di folder 'assets/')
team_members = [
    {"nama": "MHD. HASBI", "nim": "2311522032", "foto": "assets/hasbi.jpg"},
    {"nama": "Laila Qadriyah", "nim": "2311522022", "foto": "assets/laila.jpg"},
    {"nama": "Raihanah Alya Rahmadi", "nim": "2311522016", "foto": "assets/raihanah.jpg"},
]

# --- FUNGSI BASE64 UNTUK EMBED GAMBAR (MEMPERBAIKI GAMBAR TIDAK MUNCUL) ---
@st.cache_data
def get_image_base64(path):
    """Membaca gambar dan mengkonversinya ke Base64 untuk di-embed dalam HTML."""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"⚠️ File foto tidak ditemukan: {path}. Pastikan folder 'assets/' berisi file yang benar.")
        # Mengembalikan placeholder Base64 untuk gambar transparan/kosong
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# --------------------------------------------------------------------------

st.markdown(f"""
    <style>
    /* ---------------------------------------------------- */
    /* GLOBAL & RESPONSIVENESS STYLING (NEW)                */
    /* ---------------------------------------------------- */
    html, body, .main, .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    /* Responsive Title Styling */
    .title {{
        text-align: center;
        font-size: 48px; /* Default for desktop */
        font-weight: 800;
        background: linear-gradient(45deg, #155724, {GREEN_COLOR}); 
        background-size: 400% 400%; 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        animation: gradient-animation 5s ease infinite; 
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.4); 
        letter-spacing: 2px;
        margin-top: 40px;
    }}

    @media (max-width: 768px) {{
        .title {{
            font-size: 32px; /* Smaller for mobile/tablet */
            margin-top: 20px;
        }}
        .stButton > button {{
             padding: 10px 10px;
             font-size: 14px;
        }}
    }}
    /* ---------------------------------------------------- */
    /* CUSTOM COMPONENTS STYLING                            */
    /* ---------------------------------------------------- */

    /* Button Styling */
    .stButton > button {{
        width: 100%;
        border-radius: 12px;
        background-color: {GREEN_COLOR};
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .stButton > button:hover {{
        background-color: {DARK_GREEN};
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }}

    @keyframes gradient-animation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    h1 {{
        text-align: center;
        color: {DARK_GREEN} !important;
        margin-top: 40px;
        font-weight: 700;
    }}
    div.stProgress > div > div {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 99999999;
        background: linear-gradient(90deg, {GREEN_COLOR}, {DARK_GREEN}); 
        padding: 8px;
        border-radius: 5px 5px 0 0;
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
    }}
    .card-container {{
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        margin-bottom: 20px;
        background-color: {GRAY_LIGHT};
        border-left: 5px solid {GREEN_COLOR};
    }}

    /* ---------------------------------------------------- */
    /* TEAM MEMBER STYLING (UPDATED)                        */
    /* ---------------------------------------------------- */
    .member-card {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 15px;
        margin: 0; 
        border-radius: 10px;
        transition: transform 0.3s;
        min-width: 150px; 
    }}
    .member-card:hover {{
        transform: translateY(-5px);
        background-color: #e9ecef;
    }}
    .member-photo {{
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid {DARK_GREEN};
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }}
    .member-name {{
        font-weight: 700;
        color: {DARK_GREEN};
        font-size: 16px;
    }}
    .member-nim {{
        font-size: 14px;
        color: #6c757d;
    }}

    /* Hapus .team-grid karena kita akan menggunakan st.columns */
    .team-grid {{
        display: flex;
        justify-content: space-around;
        gap: 20px;
        margin-top: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# Session State Initialization (UPDATED: Menambahkan state untuk menyimpan pilihan Analisis Data)
if 'current_page_idx' not in st.session_state:
    st.session_state.current_page_idx = 0
if 'navigation' not in st.session_state:
    st.session_state.navigation = "Home"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
# State untuk mempertahankan pilihan Multiselect
if 'selected_predictors' not in st.session_state:
    st.session_state.selected_predictors = []
# State untuk mempertahankan pilihan Selectbox
if 'selected_target' not in st.session_state:
    st.session_state.selected_target = None
# State untuk menyimpan parameter C dan Gamma yang digunakan
if 'C_param_used' not in st.session_state:
    st.session_state.C_param_used = 1.0
if 'Gamma_param_used' not in st.session_state:
    st.session_state.Gamma_param_used = 0.1
# State untuk memulai proses upload data baru
if 'start_upload' not in st.session_state:
     st.session_state.start_upload = False


pages = ["Home", "Input Dataset", "Preprocessing Data", "Analisis Data", "Data Visualization", "Kesimpulan & Prediksi"]
progress_bar = st.progress(0)

# --------------------------
# 2. PAGE FUNCTIONS
# --------------------------

def display_team_member(name, nim, photo_path):
    """
    Menampilkan kartu anggota tim dengan foto melingkar.
    Menggunakan HTML/CSS dan base64 untuk mengatasi masalah path di Streamlit.
    """
    
    # Dapatkan Base64 dari gambar
    img_base64 = get_image_base64(photo_path)
    
    st.markdown(f"""
        <div class="member-card">
            <img src="data:image/jpeg;base64,{img_base64}" class="member-photo" alt="{name}">
            <div class="member-name">{name}</div>
            <div class="member-nim">NIM: {nim}</div>
        </div>
    """, unsafe_allow_html=True)


def home():
    progress_bar.progress(0)
    st.markdown(
    """
    <div class="title">
        PREDIKSI RISIKO DROPOUT MAHASISWA (SVM)
    </div>
    """, 
    unsafe_allow_html=True
)
    st.markdown(f'<div style="text-align: center; color: {DARK_GREEN}; margin-top: 10px; font-size: 20px; font-weight: 500;">Aplikasi Sederhana Klasifikasi Data Mahasiswa Berbasis Web</div>', unsafe_allow_html=True)

    st.image("assets/gawoh-11.png", caption="Aplikasi Prediksi Risiko Akademik")
    
    st.markdown("---")
    st.subheader("💡 Ringkasan Proyek")
    st.info(
        "Aplikasi ini menggunakan Algoritma **Support Vector Machine (SVM)** untuk memprediksi risiko **Dropout**, **Lulus**, atau **Masih Aktif Kuliah (Enrolled)**. Ikuti langkah-langkah di *sidebar* untuk mengolah data dan melatih model Anda."
    )
    
    # --- BAGIAN ANGGOTA KELOMPOK (UPDATED LAYOUT) ---
    st.markdown("---")
    st.subheader("👥 Anggota Kelompok Proyek")

    # Menggunakan st.columns untuk layout horizontal-ke-vertikal yang responsif
    col_hasbi, col_laila, col_raihanah = st.columns(3)
    
    with col_hasbi:
        display_team_member(team_members[0]['nama'], team_members[0]['nim'], team_members[0]['foto'])

    with col_laila:
        display_team_member(team_members[1]['nama'], team_members[1]['nim'], team_members[1]['foto'])
        
    with col_raihanah:
        display_team_member(team_members[2]['nama'], team_members[2]['nim'], team_members[2]['foto'])

    # --- END BAGIAN ANGGOTA KELOMPOK ---

    st.markdown("---")

    # NEW ADDITION: Why SVM? (REVISED TEXT)
    st.subheader("🧠 Mengapa Memilih Support Vector Machine (SVM)?")
    st.markdown("""
    Dalam tugas klasifikasi yang kompleks seperti memprediksi status akhir mahasiswa, pemilihan algoritma sangatlah krusial. Kami memilih **Support Vector Machine (SVM)** dibandingkan metode lain karena memiliki kapabilitas unik untuk menangani data multidimensi dan menemukan batas pemisah yang optimal:

    1.  **Kemampuan Menangani Batas Keputusan Non-linear (Kernel Trick):** Data mahasiswa (nilai, finansial, demografi) seringkali tidak dapat dipisahkan secara garis lurus. SVM, terutama dengan *kernel* RBF, unggul dalam menemukan *hyperplane* pemisah yang non-linear, secara efektif memisahkan risiko 'Dropout', 'Lulus', dan 'Enrolled' bahkan di ruang fitur yang kompleks. 
    2.  **Fokus pada Kasus Kritis (Support Vectors):** Alih-alih melibatkan semua data, SVM hanya bergantung pada 'Support Vectors' (titik-titik data yang paling dekat dengan batas keputusan). Hal ini membuat model **sangat efisien** dan **kuat terhadap *outlier*** (data ekstrem) yang tidak berada di dekat batas pemisah, memastikan prediksi yang stabil.
    3.  **Regularisasi yang Terkontrol:** Parameter **C** menyediakan kontrol langsung atas keseimbangan antara meminimalkan kesalahan klasifikasi pada data latih (akurasi tinggi) dan menciptakan batas keputusan yang mulus (mencegah *overfitting*). Ini penting untuk memastikan model kami dapat *menggeneralisasi* dengan baik ke data mahasiswa baru yang belum pernah dilihat sebelumnya.
    """)


def input_dataset():
    progress_bar.progress(20)
    st.markdown('<h1>📥 Input Dataset</h1>', unsafe_allow_html=True)
    
    # Keterangan Tambahan
    st.markdown("#### Tujuan Halaman")
    st.info("Halaman ini berfungsi sebagai langkah awal Akuisisi Data, di mana Anda dapat mengunggah dataset berformat CSV yang akan digunakan untuk melatih model prediksi.")

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.subheader("✅ Data Berhasil Diunggah")
        st.dataframe(data)
        st.write(f"Dataset memiliki **{data.shape[0]}** baris dan **{data.shape[1]}** kolom. Lanjutkan ke langkah *Preprocessing*.")
        # Reset data_cleaned dan pipeline agar sinkron
        if 'data_cleaned' in st.session_state:
             st.session_state.data_cleaned = None
        st.session_state.pipeline = None
        st.session_state.prediction_made = False
        st.session_state.start_upload = False # Reset upload state


def preprocessing_data():
    progress_bar.progress(40)
    if st.session_state.data is None:
        st.error("⚠️ Mohon unggah file CSV terlebih dahulu di halaman Input Dataset.")
        return

    data = st.session_state.data
    st.markdown('<h1>⚙️ Preprocessing Data</h1>', unsafe_allow_html=True)
    
    # Keterangan Tambahan
    st.markdown("#### Tujuan Halaman")
    st.info("Tahap *Preprocessing* bertujuan membersihkan dan mempersiapkan data agar sesuai untuk analisis model. Data yang kotor dapat menyebabkan hasil model yang bias atau tidak akurat.")


    # 2.1 RINGKASAN DATA
    with st.expander("🔎 Statistik & Ringkasan Data Awal", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Baris", data.shape[0])
        with col2:
            st.metric("Total Kolom", data.shape[1])
        
        st.subheader("Statistik Deskriptif Data Numerik")
        st.dataframe(data.describe().T)

        st.subheader("Jumlah Nilai Unik per Kolom Kategori")
        categorical_cols = data.select_dtypes(include=['object']).columns
        unique_counts = {col: data[col].nunique() for col in categorical_cols}
        st.dataframe(pd.DataFrame(list(unique_counts.items()), columns=['Kolom', 'Nilai Unik']))

    # 2.2 PENANGANAN MISSING VALUE & DUPLIKAT
    with st.expander("🧹 Pembersihan Data (Duplikat & Null)", expanded=True):
        num_duplicates = data.duplicated().sum()
        total_nulls = data.isnull().sum().sum()
        
        st.write(f"**Total Baris Duplikat:** {num_duplicates}")
        st.write(f"**Total Nilai Null:** {total_nulls}")

        if num_duplicates > 0 or total_nulls > 0:
            st.warning("Data mengandung duplikat atau nilai null. Direkomendasikan untuk dibersihkan.")
            
            cleaning_option = st.radio(
                "Pilih Metode Pembersihan (untuk data null):",
                ('Hapus Baris Null (Dropout)', 'Isi dengan Modus (Imputasi)'),
                index=0
            )

            if st.button('Bersihkan Data Sekarang', key='clean_btn'):
                data_cleaned = data.drop_duplicates()
                
                if cleaning_option == 'Hapus Baris Null (Dropout)':
                    data_cleaned = data_cleaned.dropna()
                    st.success("Baris duplikat dan null berhasil dihapus.")
                else:
                    for col in data_cleaned.columns:
                        if data_cleaned[col].isnull().any():
                            mode_value = data_cleaned[col].mode()[0]
                            data_cleaned[col] = data_cleaned[col].fillna(mode_value)
                    st.success("Baris duplikat dihapus, dan nilai null diisi dengan Modus.")

                st.session_state.data_cleaned = data_cleaned
                st.write(f"Baris setelah pembersihan: {data_cleaned.shape[0]}")
                st.dataframe(data_cleaned.head())
                st.session_state.pipeline = None # Reset pipeline
                st.session_state.prediction_made = False
                st.session_state.start_upload = False # Reset upload state
        else:
            st.session_state.data_cleaned = data
            st.success("Data sudah bersih. Anda dapat melanjutkan ke Analisis Data.")


def analisis_data():
    progress_bar.progress(60)
    data = st.session_state.get('data_cleaned', st.session_state.data)
    
    if data is None or ('data_cleaned' not in st.session_state and st.session_state.data is None):
        st.error("⚠️ Mohon unggah file CSV dan bersihkan data terlebih dahulu.")
        return

    st.markdown('<h1>🔬 Analisis Data (Klasifikasi SVM)</h1>', unsafe_allow_html=True)
    
    # NEW ADDITION: Guidance on Variable Selection 
    with st.expander("❓ Bagaimana Menentukan Variabel Prediktor Terbaik?", expanded=True):
        st.subheader("Strategi Pemilihan Fitur (Feature Selection)")
        st.markdown("""
        Penentuan variabel prediktor (fitur) sangat krusial karena memengaruhi kinerja model. Dalam konteks prediksi risiko akademik (dropout), pemilihan variabel harus didasarkan pada:
        
        1.  **Domain Knowledge (Pengetahuan Domain):** Pilih variabel yang secara logis berkorelasi dengan kinerja atau motivasi mahasiswa (misalnya, nilai, status finansial, dan keterlibatan akademik).
        2.  **Hasil Akademik Awal:** Kinerja mahasiswa di semester-semester awal sering kali menjadi prediktor yang paling kuat untuk status akhir.
        """)
        
        best_predictors_suggestion = [
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Admission grade",
            "Tuition fees up to date",
            "Debtor",
            "Scholarship holder",
            "Age at enrollment"
        ]
        
        st.markdown(f"""
        **Saran Variabel Prediktor Terbaik:**
        Berdasarkan studi kasus umum dan variabel dalam dataset Anda, variabel yang paling berpengaruh terhadap risiko akademik meliputi:
        
        - **Kinerja Akademik:** `{best_predictors_suggestion[0]}` (Jumlah mata kuliah yang disetujui di semester 1), `{best_predictors_suggestion[1]}` (Semester 2), dan `{best_predictors_suggestion[2]}` (Nilai rata-rata Semester 2).
        - **Finansial/Administrasi:** `{best_predictors_suggestion[4]}`, `{best_predictors_suggestion[5]}`, dan `{best_predictors_suggestion[6]}`.
        - **Latar Belakang:** `{best_predictors_suggestion[3]}` dan `{best_predictors_suggestion[7]}`.
        
        **Rekomendasi:** Untuk hasil optimal, cobalah untuk memilih minimal 5 dari variabel yang disebutkan di atas.
        """)
    # END NEW ADDITION

    with st.expander("📝 Konfigurasi Model", expanded=True):
        col_select, col_target = st.columns(2)
        column_names = data.columns.tolist()

        # 1. Pemilihan Variabel Prediktor
        with col_select:
            prediction_variable = st.multiselect(
                "Pilih Variabel Prediktor (X) - Minimal 2:", 
                column_names,
                default=st.session_state.selected_predictors,
                key='predictor_select'
            )
            st.session_state.selected_predictors = prediction_variable
        
        # 2. Pemilihan Variabel Target
        with col_target:
            default_target = [col for col in column_names if col in ['Target', 'target']]
            if st.session_state.selected_target and st.session_state.selected_target in column_names:
                default_index = column_names.index(st.session_state.selected_target)
            elif default_target:
                default_index = column_names.index(default_target[0])
            else:
                default_index = len(column_names) - 1

            target_variable = st.selectbox(
                "Pilih Variabel Target (Y):", 
                column_names, 
                index=default_index,
                key='target_select'
            )
            st.session_state.selected_target = target_variable


        st.subheader("💡 Penjelasan: Hyperparameter Tuning SVC")
        st.info("""
            Hyperparameter C dan Gamma sangat mempengaruhi hasil model:
            - **C (Regularization):** Mengontrol toleransi model terhadap kesalahan klasifikasi. Nilai C yang **besar** cenderung menyebabkan *overfitting*, sedangkan nilai C **kecil** menghasilkan batas keputusan yang lebih umum.
            - **Gamma (Kernel Coeff.):** Mendefinisikan seberapa besar pengaruh satu sampel data pelatihan. Gamma **besar** berarti pengaruh lokal (batas keputusan kompleks), Gamma **kecil** berarti pengaruh global (batas keputusan mulus).
        """)
        
        st.subheader("✨ Tuning Hyperparameter SVC")
        col_c, col_gamma = st.columns(2)

        with col_c:
            C_val = st.slider('Parameter C (Regularization Strength)', 0.1, 10.0, st.session_state.C_param_used, 0.1, key='C_slider')
            
        with col_gamma:
            gamma_val = st.slider('Parameter Gamma (Kernel Coefficient)', 0.001, 1.0, st.session_state.Gamma_param_used, 0.001, key='gamma_slider')


    # Pastikan pelatihan model menggunakan nilai yang sudah tersinkronisasi
    C_to_use = st.session_state.C_slider
    gamma_to_use = st.session_state.gamma_slider


    if len(st.session_state.selected_predictors) >= 2 and st.session_state.selected_target:
        if st.button('🚀 Latih Model SVM', key='train_btn'):
            prediction_variable = st.session_state.selected_predictors
            target_variable = st.session_state.selected_target
            
            # Preprocessing dan Latihan Model
            X = data[prediction_variable]
            y = data[target_variable]
            
            # Label Encoding untuk kolom objek (jika ada) di Prediktor
            X = pd.get_dummies(X, drop_first=True)
            
            # Label Encoding untuk Target
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_labels = le.classes_

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',  
                    C=C_to_use, 
                    gamma=gamma_to_use, 
                    random_state=42,
                    probability=True
                ))
            ])

            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, 'model_dropout_svc.pkl')
            
            # Simpan semua hasil dan objek (TERMASUK PARAMETER C & GAMMA YANG DIGUNAKAN)
            st.session_state.pipeline = pipeline
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.le = le
            st.session_state.prediction_variable = prediction_variable
            st.session_state.target_variable = target_variable
            st.session_state.class_labels = class_labels
            st.session_state.accuracy = accuracy_score(y_test, pipeline.predict(X_test))
            st.session_state.classification_report = classification_report(y_test, pipeline.predict(X_test), target_names=class_labels, output_dict=True)
            st.session_state.confusion_matrix = confusion_matrix(y_test, pipeline.predict(X_test))
            st.session_state.C_param_used = C_to_use # Simpan nilai yang digunakan
            st.session_state.Gamma_param_used = gamma_to_use # Simpan nilai yang digunakan
            st.session_state.prediction_made = False
            st.session_state.start_upload = False # Reset upload state

            st.success(f"Model SVM (C={C_to_use}, Gamma={gamma_to_use}) telah dilatih dan disimpan.")
            st.markdown(f"**Akurasi Model:** <span style='color:{DARK_GREEN}; font-size: 24px;'>**{st.session_state.accuracy * 100:.2f}%**</span>", unsafe_allow_html=True)
            
        elif st.session_state.pipeline is not None and st.session_state.C_param_used == C_to_use and st.session_state.Gamma_param_used == gamma_to_use:
             st.success(f"Model sudah dilatih (C={C_to_use}, Gamma={gamma_to_use}) dan siap digunakan. Anda dapat mengatur ulang parameter jika ingin melatih kembali.")
        elif st.session_state.pipeline is not None and (st.session_state.C_param_used != C_to_use or st.session_state.Gamma_param_used != gamma_to_use):
             st.warning(f"Parameter telah diubah (C: {st.session_state.C_param_used} -> {C_to_use}, Gamma: {st.session_state.Gamma_param_used} -> {gamma_to_use}). Tekan 'Latih Model SVM' untuk mengimplementasikan perubahan.")
        
    else:
        st.error("⚠️ Mohon pilih minimal 2 Variabel Prediktor dan 1 Variabel Target untuk melanjutkan.")


def visualisasi_data():
    progress_bar.progress(80)
    if st.session_state.pipeline is None:
        st.error("⚠️ Model belum dilatih. Silakan lakukan analisis data terlebih dahulu.")
        return

    pipeline = st.session_state.pipeline
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    le = st.session_state.le
    class_labels = st.session_state.class_labels

    st.markdown('<h1>📊 Data Visualization</h1>', unsafe_allow_html=True)
    
    # Keterangan Tambahan
    st.markdown("#### Tujuan Halaman")
    st.info("Halaman ini menyajikan visualisasi hasil prediksi model SVC Anda pada data uji (test data) untuk mempermudah interpretasi performa.")

    st.write("### Perbandingan Prediksi vs Aktual")
    y_pred = pipeline.predict(X_test)

    comparison_df = pd.DataFrame({
        'Aktual': le.inverse_transform(y_test),
        'Prediksi': le.inverse_transform(y_pred)
    })

    comparison_df['Hasil'] = np.where(comparison_df['Aktual'] == comparison_df['Prediksi'], 'Benar', 'Salah')

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = px.pie(
            comparison_df,
            names='Hasil',
            title="Distribusi Akurasi Prediksi",
            color_discrete_map={'Benar': GREEN_COLOR, 'Salah': '#B80000'}
        )
        st.plotly_chart(fig_pie, use_container_width=True) # Tambahkan use_container_width
    
    with col_bar:
        prediction_distribution = comparison_df['Prediksi'].value_counts().reset_index()
        prediction_distribution.columns = ['Status', 'Jumlah']
        
        fig_bar = px.bar(
            prediction_distribution,
            x='Status',
            y='Jumlah',
            title='Distribusi Prediksi Kelas',
            color='Status',
            color_discrete_map={label: GREEN_COLOR if label != 'Dropout' else '#B80000' for label in class_labels}
        )
        st.plotly_chart(fig_bar, use_container_width=True) # Tambahkan use_container_width

    correct_predictions = (y_test == y_pred).sum()
    total_predictions = len(y_test)
    st.markdown(f"**Prediksi Benar:** <span style='color:{DARK_GREEN}; font-weight: bold;'>{correct_predictions}</span> dari {total_predictions} data uji.", unsafe_allow_html=True)


def kesimpulan_dan_prediksi():
    progress_bar.progress(100)
    st.markdown('<h1>📜 Kesimpulan & Prediksi Data Baru</h1>', unsafe_allow_html=True)

    if st.session_state.pipeline is None:
        st.error("⚠️ Model belum dilatih. Silakan kembali ke Analisis Data.")
        return

    # --- BAGIAN 1: KESIMPULAN MODEL ---
    st.subheader("1. Ringkasan Performa Model (Data Uji)")
    
    col_acc, col_rep = st.columns(2)
    with col_acc:
        st.metric("Akurasi Model SVC", f"{st.session_state.accuracy * 100:.2f}%")
        
        # Confusion Matrix
        cm = st.session_state.confusion_matrix
        class_labels = st.session_state.class_labels
        
        # Inisialisasi figure (penting untuk Streamlit agar tidak menumpuk plot)
        fig, ax = plt.subplots(figsize=(6, 5)) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=14)
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
        
        st.pyplot(fig) # Tampilkan figure
        plt.close(fig) # Tutup figure untuk membebaskan memori

    with col_rep:
        st.write("Classification Report (Precision, Recall, F1-Score):")
        report_df = pd.DataFrame(st.session_state.classification_report).transpose().round(2)
        st.dataframe(report_df)
    
    st.markdown("---")

    # --- BAGIAN 2: ANALISIS MENDALAM (NEW INSIGHTS) ---
    st.subheader("2. Analisis Mendalam Konfigurasi & Hasil Model")
    
    st.markdown("#### Variabel Prediktor yang Digunakan:")
    if st.session_state.prediction_variable:
        st.code(", ".join(st.session_state.prediction_variable))
    
    st.markdown(f"""
    Model SVC ini dilatih untuk memprediksi **{st.session_state.target_variable}** menggunakan **{len(st.session_state.prediction_variable)}** variabel prediktor yang Anda pilih. Pemilihan variabel ini secara langsung memengaruhi batas keputusan yang dibuat oleh model.
    """)
    
    st.markdown("#### Dampak Parameter SVC pada Pelatihan:")
    C_used = st.session_state.C_param_used
    Gamma_used = st.session_state.Gamma_param_used
    
    # NEW INSIGHT 1: Interpretation of Hyperparameters
    st.markdown(f"""
    - **Parameter C (Regularization): {C_used}**: Nilai C yang digunakan ini [**menentukan tingkat hukuman**] terhadap setiap kesalahan klasifikasi. Dengan nilai **{C_used}**, model mencoba menyeimbangkan antara membatasi kesalahan pada data latih dan mempertahankan batas keputusan yang cukup umum (*generalized*).
    - **Parameter Gamma (Kernel Coeff.): {Gamma_used}**: Nilai Gamma ini [**mendefinisikan pengaruh**] satu sampel data. Dengan Gamma={Gamma_used}, model akan membuat keputusan berdasarkan pengaruh yang [**bersifat global**] (karena nilai cukup kecil) terhadap batas keputusan, menciptakan batas pemisah yang relatif mulus.
    """)
    
    # NEW INSIGHT 2: Interpretation of Critical Metrics (Point 2)
    report = st.session_state.classification_report
    accuracy = st.session_state.accuracy
    
    if any("Dropout" in label for label in st.session_state.class_labels):
        dropout_key = next((key for key in report.keys() if "Dropout" in key), None)
        if dropout_key:
            dropout_recall = report[dropout_key].get('recall', 0)
            dropout_precision = report[dropout_key].get('precision', 0)
        else:
            dropout_recall = 0
            dropout_precision = 0
    else:
        dropout_recall = 0
        dropout_precision = 0
        
    recall_display = f"{dropout_recall * 100:.0f}%" if dropout_recall > 0 else "N/A"
    precision_display = f"{dropout_precision * 100:.0f}%" if dropout_precision > 0 else "N/A"
    
    st.markdown("#### Interpretasi Hasil Kritis (Relevansi Topik Dropout)")
    st.markdown(f"""
    - **Insight Utama (Recall Dropout {recall_display})**: Angka Recall untuk kelas yang terkait dengan **Dropout** sangat penting. Nilai **{recall_display}** menunjukkan bahwa model berhasil mengidentifikasi mahasiswa yang berisiko dropout dari total kasus dropout yang sebenarnya. **Tujuan utama sistem ini adalah meminimalkan *False Negative*** (mahasiswa berisiko tinggi terlewatkan), dan Recall adalah metrik kuncinya.
    - **Keandalan Peringatan (Precision Dropout {precision_display})**: Presisi **{precision_display}** berarti dari semua mahasiswa yang diprediksi model sebagai 'Dropout', sebagian besar prediksi tersebut benar. Ini mengukur seberapa andal (tepat) peringatan risiko yang diberikan.
    - **Kesimpulan Terkait Topik**: Akurasi keseluruhan model sebesar **{accuracy * 100:.2f}%** menunjukkan bahwa model SVM Anda **efektif** sebagai alat skrining awal untuk intervensi akademik. Fokus pada pengoptimalan **Recall Dropout** harus menjadi prioritas untuk implementasi di dunia nyata, memungkinkan institusi untuk membantu mahasiswa tepat waktu.
    """)
    
    st.markdown("---")


    # --- BAGIAN 3: PREDIKSI DATA BARU ---
    st.subheader("3. Prediksi Data Baru")
    
    st.markdown("Tahap ini memungkinkan Anda menguji model yang telah dilatih dengan data mahasiswa baru untuk mendapatkan prediksi status akademik mereka.")
    
    # Tombol konfirmasi sebelum mengupload file
    if st.button('Mulai Unggah File untuk Prediksi', key='start_upload_btn') or st.session_state.get('prediction_made', False):
        st.session_state.start_upload = True

    if st.session_state.get('start_upload', False):
        try:
            # Pengecekan kolom yang digunakan dalam model
            if 'X_test' not in st.session_state:
                 st.error("⚠️ Data latih awal (X_test) tidak ditemukan. Mohon latih ulang model.")
                 return
                 
            loaded_model = st.session_state.pipeline 
            
            st.info("File CSV yang Anda unggah akan diprediksi status mahasiswanya. Pastikan file hanya berisi variabel prediktor yang telah dipilih.")

            uploaded_file_new = st.file_uploader("Upload file CSV data baru (Inferensi)", type=["csv"], key='new_upload')

            if uploaded_file_new is not None:
                data_baru = pd.read_csv(uploaded_file_new)
                prediction_variable = st.session_state.prediction_variable
                le = st.session_state.le
                
                if set(prediction_variable).issubset(data_baru.columns):
                    
                    # Logika Prediksi
                    X_baru = data_baru[prediction_variable].copy() # Gunakan copy untuk menghindari SettingWithCopyWarning
                    X_baru = pd.get_dummies(X_baru, drop_first=True)

                    # Menyelaraskan kolom dengan data latih (penting untuk OHE)
                    X_test_cols = st.session_state.X_test.columns.tolist()
                    X_baru_aligned = pd.DataFrame(0, index=X_baru.index, columns=X_test_cols)
                    
                    # Isi kolom yang ada di X_baru ke X_baru_aligned
                    common_cols = list(set(X_baru.columns) & set(X_test_cols))
                    X_baru_aligned[common_cols] = X_baru[common_cols]
                    
                    # Scaling dan Prediksi
                    scaler = loaded_model.named_steps['scaler']
                    classifier = loaded_model.named_steps['classifier']
                    
                    X_baru_scaled = scaler.transform(X_baru_aligned)
                    y_pred_baru = classifier.predict(X_baru_scaled)
                    
                    y_pred_baru_labels = le.inverse_transform(y_pred_baru)
                    data_baru['Prediksi_Status'] = y_pred_baru_labels

                    st.session_state.prediction_made = True
                    
                    # Tampilkan Hasil Prediksi
                    st.subheader("✅ Hasil Prediksi")
                    pred_count = data_baru['Prediksi_Status'].value_counts()
                    
                    st.markdown(f"**Total Data yang Diprediksi:** **{data_baru.shape[0]}**")
                    for label, count in pred_count.items():
                          st.write(f"  - Prediksi **{label}**: **{count}** siswa")

                    st.subheader("Tabel Data Lengkap dengan Hasil Prediksi")
                    st.dataframe(data_baru)
                else:
                    missing_cols = set(prediction_variable) - set(data_baru.columns)
                    st.error(f"⚠️ Kolom data baru tidak sesuai. Variabel prediktor yang hilang: {', '.join(missing_cols)}. Pastikan semua kolom yang Anda pilih sebelumnya ada di file baru.")
            
        except FileNotFoundError:
             st.error("⚠️ File model ('model_dropout_svc.pkl') tidak dapat dimuat. Mohon latih model di halaman 'Analisis Data' terlebih dahulu.")
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat memproses data: {e}. Pastikan data baru memiliki format yang sesuai.")


# --------------------------
# 3. MAIN APP STRUCTURE & NAVIGASI GUARDRAIL
# --------------------------

def check_page_completion(page_idx):
    """Memeriksa apakah persyaratan halaman saat ini sudah dipenuhi."""
    # Index 1: Input Dataset - Data harus ada
    if page_idx == 1: 
        return st.session_state.data is not None
    # Index 2: Preprocessing Data - Data bersih harus ada
    if page_idx == 2:
        # Cek data bersih, jika belum ada, pakai data awal dan beri peringatan
        if 'data_cleaned' not in st.session_state:
             return st.session_state.data is not None
        return st.session_state.get('data_cleaned') is not None
    # Index 3: Analisis Data - Model harus sudah dilatih
    elif page_idx == 3: 
        # Cek apakah sudah ada model yang tersimpan di state
        return st.session_state.pipeline is not None
    # Index 4: Data Visualization - Model harus sudah dilatih
    elif page_idx == 4: 
        return st.session_state.pipeline is not None
    # Index 5: Kesimpulan & Prediksi - Cukup memastikan model sudah ada
    elif page_idx == 5: 
        return st.session_state.pipeline is not None
    return True 


def main():
    
    def next_page():
        if check_page_completion(st.session_state.current_page_idx) and st.session_state.current_page_idx < len(pages) - 1:
            st.session_state.current_page_idx += 1
            st.session_state.navigation = pages[st.session_state.current_page_idx]
            
    def prev_page():
        if st.session_state.current_page_idx > 0:
            st.session_state.current_page_idx -= 1
            st.session_state.navigation = pages[st.session_state.current_page_idx]
    
    st.sidebar.title("📚 Navigasi Proyek")
    selected_page = st.sidebar.selectbox(
        "Pilih Halaman", 
        pages, 
        key='navigation',
        index=st.session_state.current_page_idx
    )
    
    st.session_state.current_page_idx = pages.index(selected_page)

    st.sidebar.markdown(
        """
        <div style="position: fixed; bottom: 30px;left: 45px; width: 100%; font-size: 16px; color: #555; text-align: left;">
            Made with Streamlit & Python
        </div>
        """, 
        unsafe_allow_html=True
    )  
    
    current_page = pages[st.session_state.current_page_idx]
    if current_page == "Home":
        home()
    elif current_page == "Input Dataset":
        input_dataset()
    elif current_page == "Preprocessing Data":
        preprocessing_data()
    elif current_page == "Analisis Data":
        analisis_data()
    elif current_page == "Data Visualization":
        visualisasi_data()
    elif current_page == "Kesimpulan & Prediksi":
        kesimpulan_dan_prediksi()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_page_idx > 0:
            st.button('⬅️ Previous', on_click=prev_page, key='prev_btn')
    
    with col3:
        can_proceed = check_page_completion(st.session_state.current_page_idx)
        if st.session_state.current_page_idx < len(pages) - 1:
            if can_proceed:
                st.button('Next ➡️', on_click=next_page, key='next_btn')
            else:
                st.button('Next ➡️', disabled=True, key='next_btn_disabled')
                
    
    if st.session_state.current_page_idx < len(pages) - 1 and not can_proceed:
        st.warning("⚠️ Penuhi persyaratan halaman ini untuk melanjutkan. (Misal: Unggah Data, Bersihkan Data, atau Latih Model)")


if __name__ == "__main__":
    # Peringatan untuk pengguna agar memastikan file gambar ada
    if not Path("assets/hasbi.jpg").is_file() or not Path("assets/laila.jpg").is_file() or not Path("assets/raihanah.jpg").is_file():
        st.error("Pastikan file 'hasbi.jpg', 'laila.jpg', dan 'raihanah.jpg' berada di dalam folder 'assets/' untuk menampilkan foto anggota kelompok.")
        
    main()