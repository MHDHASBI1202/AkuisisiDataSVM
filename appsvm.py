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

st.markdown(f"""
    <style>
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
    /* Title Styling (Home Page) */
    .title {{
        text-align: center;
        font-size: 48px;
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
        background-color: #f9f9f9;
        border-left: 5px solid {GREEN_COLOR};
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


pages = ["Home", "Input Dataset", "Preprocessing Data", "Analisis Data", "Data Visualization", "Kesimpulan & Prediksi"]
progress_bar = st.progress(0)

# --------------------------
# 2. PAGE FUNCTIONS
# --------------------------

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
    
    # NEW ADDITION: Why SVM?
    st.subheader("🧠 Mengapa Menggunakan Support Vector Machine (SVM)?")
    st.markdown("""
    Kami memilih algoritma **Support Vector Machine (SVM)** dibandingkan metode lain (seperti Decision Tree atau Naive Bayes) karena beberapa keunggulan kunci yang relevan dalam masalah klasifikasi risiko akademik:
    
    1.  **Efektif di Ruang Dimensi Tinggi:** SVM sangat baik dalam memproses data dengan banyak fitur (variabel prediktor), seperti data mahasiswa yang sering kali memiliki banyak atribut demografi dan akademik.
    2.  **Batas Keputusan yang Jelas (Hyperplane):** SVM bertujuan mencari *hyperplane* optimal yang memiliki margin pemisah terbesar antara kelas-kelas. Dengan menggunakan *kernel* (seperti RBF yang digunakan di sini), SVM dapat menciptakan batas keputusan yang non-linear dan sangat efektif, memisahkan kelas 'Dropout', 'Lulus', dan 'Enrolled' secara optimal.
    3.  **Regularisasi Inherent:** Parameter **C** dalam SVM secara alami melakukan regularisasi. Ini membantu menyeimbangkan antara mencapai akurasi tinggi pada data pelatihan (meminimalkan kesalahan) dan menjaga batas keputusan tetap mulus (mencegah *overfitting*).
    4.  **Kinerja Solid pada Data Kompleks:** SVM cenderung memberikan kinerja yang kuat pada masalah klasifikasi yang kompleks dan non-linear, seperti memprediksi hasil akhir dari variabel yang saling berkaitan.
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
        st.session_state.data_cleaned = None
        st.session_state.pipeline = None
        st.session_state.prediction_made = False


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
            
            X = pd.get_dummies(X, drop_first=True)
            
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

            st.success(f"Model SVM (C={C_to_use}, Gamma={gamma_to_use}) telah dilatih dan disimpan.")
            st.markdown(f"**Akurasi Model:** <span style='color:{DARK_GREEN}; font-size: 24px;'>**{st.session_state.accuracy * 100:.2f}%**</span>", unsafe_allow_html=True)
            
        elif st.session_state.pipeline is not None:
            st.success("Model sudah dilatih dan siap digunakan. Anda dapat mengatur ulang parameter jika ingin melatih kembali.")
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
        st.plotly_chart(fig_pie)
    
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
        st.plotly_chart(fig_bar)

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
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        st.pyplot(plt)

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
    - **Parameter Gamma (Kernel Coeff.): {Gamma_used}**: Nilai Gamma ini [**mendefinisikan pengaruh**] satu sampel data. Dengan Gamma={Gamma_used}, model akan membuat keputusan berdasarkan pengaruh yang [**bersifat lokal**] (jika nilai besar) atau [**global**] (jika nilai kecil) terhadap batas keputusan.
    """)
    
    # NEW INSIGHT 2: Interpretation of Critical Metrics (Point 2)
    st.markdown("#### Interpretasi Hasil Kritis (Relevansi Topik Dropout)")
    report = st.session_state.classification_report
    accuracy = st.session_state.accuracy
    
    # Safety check for 'Dropout' class
    if 'Dropout' in report:
        dropout_recall = report['Dropout'].get('recall', 0)
        dropout_precision = report['Dropout'].get('precision', 0)
    else:
        dropout_recall = 0
        dropout_precision = 0
        
    st.markdown(f"""
    - **Insight Utama (Recall Dropout {dropout_recall:.2f})**: Angka Recall untuk kelas **Dropout** sangat penting. Nilai **{dropout_recall * 100:.0f}%** menunjukkan bahwa model berhasil mengidentifikasi mahasiswa yang berisiko dropout dari total kasus dropout yang sebenarnya. **Tujuan utama sistem ini adalah meminimalkan *False Negative*** (mahasiswa berisiko tinggi terlewatkan), dan Recall adalah metrik kuncinya.
    - **Keandalan Peringatan (Precision Dropout {dropout_precision:.2f})**: Presisi **{dropout_precision * 100:.0f}%** berarti dari semua mahasiswa yang diprediksi model sebagai 'Dropout', sebagian besar prediksi tersebut benar. Ini mengukur seberapa andal (tepat) peringatan risiko yang diberikan.
    - **Kesimpulan Terkait Topik**: Akurasi keseluruhan model sebesar **{accuracy * 100:.2f}%** menunjukkan bahwa model SVM Anda **efektif** sebagai alat skrining awal untuk intervensi akademik. Fokus pada pengoptimalan **Recall Dropout** harus menjadi prioritas untuk implementasi di dunia nyata, memungkinkan institusi untuk membantu mahasiswa tepat waktu.
    """)
    
    st.markdown("---")


    # --- BAGIAN 3: PREDIKSI DATA BARU ---
    st.subheader("3. Prediksi Data Baru")
    
    st.markdown("Tahap ini memungkinkan Anda menguji model yang telah dilatih dengan data mahasiswa baru untuk mendapatkan prediksi status akademik mereka.")
    
    # Tombol konfirmasi sebelum mengupload file
    if st.button('Mulai Unggah File untuk Prediksi', key='start_upload_btn'):
        st.session_state.start_upload = True

    if st.session_state.get('start_upload', False):
        try:
            # Assume st.session_state.pipeline is available and valid after training
            loaded_model = st.session_state.pipeline 
            
            st.info("File CSV yang Anda unggah akan diprediksi status mahasiswanya.")

            uploaded_file_new = st.file_uploader("Upload file CSV data baru (Inferensi)", type=["csv"], key='new_upload')

            if uploaded_file_new is not None:
                data_baru = pd.read_csv(uploaded_file_new)
                prediction_variable = st.session_state.prediction_variable
                le = st.session_state.le
                
                if set(prediction_variable).issubset(data_baru.columns):
                    
                    # Logika Prediksi
                    X_baru = data_baru[prediction_variable]
                    X_baru = pd.get_dummies(X_baru, drop_first=True)

                    X_test_cols = st.session_state.X_test.columns.tolist()
                    for col in X_test_cols:
                        if col not in X_baru.columns:
                            X_baru[col] = 0
                    X_baru = X_baru[X_test_cols]

                    scaler = loaded_model.named_steps['scaler']
                    classifier = loaded_model.named_steps['classifier']
                    
                    X_baru_scaled = scaler.transform(X_baru)
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
                    st.error("⚠️ Kolom data baru tidak sesuai dengan variabel prediktor model. Pastikan semua kolom yang Anda pilih sebelumnya ada di file baru.")
            
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
    # Index 3: Analisis Data - Model harus sudah dilatih
    elif page_idx == 3: 
        # Cek apakah sudah ada model yang tersimpan di state
        return st.session_state.pipeline is not None
    # Index 5: Kesimpulan & Prediksi - Cukup memastikan model sudah ada
    elif page_idx == 5: 
        return st.session_state.pipeline is not None
    return True 


def main():
    
    def next_page():
        if st.session_state.current_page_idx < len(pages) - 1:
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
                st.warning("Penuhi persyaratan halaman ini untuk lanjut.")

if __name__ == "__main__":
    main()