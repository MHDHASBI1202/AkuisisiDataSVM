import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib  # Untuk menyimpan dan memuat model

# Styling the button
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background-color: #007BFF;  /* Blue color */
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
    }
    
    .stButton > button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
        transform: translateY(-3px);  /* Lift the button on hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);  /* Stronger shadow on hover */
    }

    .stButton > button:active {
        background-color: #003d7a;  /* Even darker blue when clicked */
        transform: translateY(1px);  /* Slightly pressed effect */
    }
    </style>
""", unsafe_allow_html=True)

# Menyimpan status halaman saat ini dalam session_state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0  # Halaman pertama (Index 0)

# Daftar halaman yang akan ditampilkan
pages = ["Input Dataset", "Preprocessing Data", "Analisis Data", "Data Visualization", "Use Model"]


def home():
    st.markdown(
    """
    <style>
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(45deg, #6a11cb, #2575fc); /* Gradasi biru dan ungu */
        background-size: 400% 400%; /* Efek gradasi yang dinamis */
        color: blue;
        -webkit-background-clip: text; /* Gradasi hanya pada teks */
        animation: gradient-animation 5s ease infinite; /* Animasi gradasi bergerak */
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.4); /* Bayangan teks dengan efek 3D */
        letter-spacing: 2px; /* Jarak antar huruf */
        padding: 20px 0;
        margin-top: 40px;
    }

    .progress-bar-fixed {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.9); /* Transparansi latar belakang */
        padding: 10px 0; /* Padding atas dan bawah */
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1); /* Bayangan untuk tampilan lebih menarik */
        z-index: 100;
        text-align: center;
    }
    </style>
    <div class="title">
        🔥PASS or FAIL-Prediction🔥
    </div>
    """, 
    unsafe_allow_html=True
)

    # Menampilkan gambar
    st.image("assets/gawoh-11.jpg", caption="Suasana Belajar Siswa")


st.markdown("""
    <style>
    /* Mengatur posisi container untuk progress bar */
    div.stProgress > div > div {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 99999999;  /* Menaikkan z-index agar di atas sidebar */
        background-color: #3498db;  /* Warna biru lebih elegan */
        padding: 8px;
        border-radius: 5px 5px 0 0;  /* Sudut yang membulat */
        box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1); /* Bayangan halus */
        transition: width 0.3s ease, background-color 0.3s ease; /* Transisi halus */
    }

    /* Memberikan margin bottom pada main content */
    .main {
        margin-bottom: 60px;
    }

    /* Memastikan progress bar selalu di atas sidebar */
    .stProgress {
        position: relative;
        z-index: 99999999;
    }

    /* Style untuk judul */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 600;
        color: #2c3e50; /* Warna gelap untuk judul */
        margin-top: 40px;
        text-transform: capitalize; /* Huruf kapital untuk bagian judul */
    }

    /* Menyembunyikan overflow di body */
    body {
        overflow-x: hidden;
    }

    /* Memastikan sidebar tidak menutupi progress bar */
    .css-1d391kg, .css-1q1n0ol {
        z-index: 99999998;
    }

    /* Memberikan efek gradient halus */
    div.stProgress > div > div {
        background: linear-gradient(90deg, #3498db, #2980b9); /* Gradient biru */
    }

    </style>
    """, unsafe_allow_html=True)

# Membuat progress bar
progress_bar = st.progress(0)


# Fungsi untuk halaman Input Dataset
def input_dataset():
    progress_bar.progress(25)
    # Menambahkan judul dengan style
    st.markdown('<h1 style="text-align: center; color: #1f4287;">📊 Input Dataset</h1>', unsafe_allow_html=True)

    # Container untuk file uploader dengan CSS yang dimodifikasi
    st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Membaca CSV yang di-upload
        data = pd.read_csv(uploaded_file)

        # Menampilkan preview data
        st.subheader("Preview Data")
        st.dataframe(data)  # Menampilkan 5 baris pertama

        # Menyimpan data untuk digunakan di halaman selanjutnya
        st.session_state.data = data


def preprocessing_data():
    progress_bar.progress(50)
    if 'data' not in st.session_state:
        st.error("Please upload a CSV file first.")
        return

    # Mengambil data yang sudah di-upload
    data = st.session_state.data
    
    # Menampilkan judul halaman
    st.markdown('<h1 style="text-align: center; color: #1f4287;">⚙️ Preprocessing Data</h1>', unsafe_allow_html=True)

    # Menampilkan jumlah baris dan kolom
    num_rows, num_columns = data.shape
    st.write(f"Jumlah Baris: {num_rows}")
    st.write(f"Jumlah Kolom: {num_columns}")

    # Menghitung jumlah baris duplikat
    num_duplicates = data.duplicated().sum()
    st.write(f"Jumlah Baris Duplikat: {num_duplicates}")

    # Memeriksa apakah ada baris duplikat
    if num_duplicates > 0:
        st.warning("Data masih mengandung duplikat ataupun null. Harap perbaiki terlebih dahulu.")
        st.subheader("Baris Duplikat:")
        st.dataframe(data[data.duplicated()])
    else:
        st.write("Tidak ada duplikat pada data.")

    # Cek adanya nilai null di dataset
    null_counts = data.isnull().sum()  # Menghitung jumlah nilai null per kolom
    total_nulls = null_counts.sum()  # Menghitung total nilai null dalam dataset
    
    st.write(f"Jumlah Total Nilai Null: {total_nulls}")
    
    # Menampilkan kolom dengan nilai null
    if total_nulls > 0:
        st.subheader("Kolom dengan Nilai Null:")
        st.dataframe(null_counts[null_counts > 0].sort_values(ascending=False))
    else:
        st.write("Tidak ada nilai null pada data.")

    # Memeriksa apakah ada duplikat atau null
    if num_duplicates > 0 or total_nulls > 0:
        if st.button('Hapus Duplikat dan Nilai Null'):
            # Menghapus duplikat
            data_cleaned = data.drop_duplicates()
    
            # Menghapus nilai null
            data_cleaned = data_cleaned.dropna()

            # Simpan data_cleaned ke dalam session_state
            st.session_state.data_cleaned = data_cleaned

            # Menampilkan data setelah duplikat dan nilai null dihapus
            st.subheader("Data Setelah Duplikat dan Nilai Null Dihapus")
            st.dataframe(data_cleaned)

            # Menampilkan jumlah baris dan kolom setelah pembersihan
            num_rows, num_columns = data_cleaned.shape
            st.write(f"Jumlah Baris: {num_rows}")
            st.write(f"Jumlah Kolom: {num_columns}")

            st.success("Duplikat dan Nilai Null berhasil dihapus.")

    else:
        # Jika tidak ada duplikat dan null, tampilkan pesan boleh lanjut
        st.success("Data Anda sudah bersih. Anda dapat melanjutkan ke langkah berikutnya.")



def analisis_data():
    progress_bar.progress(75)
    # Gunakan data_cleaned jika tersedia
    if 'data_cleaned' in st.session_state:
        data = st.session_state.data_cleaned
    elif 'data' in st.session_state:
        data = st.session_state.data
    else:
        st.error("Please upload a CSV file first and preprocess the data.")
        return

    st.markdown('<h1 style="text-align: center; color: #1f4287;">📝 Data Analysis</h1>', unsafe_allow_html=True)

    column_names = data.columns.tolist()
    prediction_variable = st.multiselect("Pilih Variabel Prediktor (minimal 2 variabel):", column_names)
    
    target_variable = st.selectbox("Pilih Variabel Target:", column_names)

    if len(prediction_variable) >= 2 and target_variable:
        X = data[prediction_variable]
        y = data[target_variable]

        X = pd.get_dummies(X, drop_first=True)
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, 'model_pass_fail.pkl')
        st.success("Model telah disimpan dengan nama 'model_pass_fail.pkl'.")

        # Menyimpan objek pipeline, data, dan variabel ke dalam st.session_state
        st.session_state.pipeline = pipeline
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.le = le
        st.session_state.prediction_variable = prediction_variable  # Menyimpan variabel prediktor
        st.session_state.target_variable = target_variable  # Menyimpan variabel target

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi Model: {accuracy * 100:.2f}%")



# Fungsi untuk visualisasi data
def visualisasi_data():
    progress_bar.progress(100)
    if 'data' not in st.session_state:
        st.error("Please upload a CSV file first.")
        return

    if 'pipeline' not in st.session_state:
        st.error("Model belum dilatih. Silakan lakukan analisis data terlebih dahulu.")
        return

    pipeline = st.session_state.pipeline
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    le = st.session_state.le

    st.markdown('<h1 style="text-align: center; color: #1f4287;">📈 Data Visualization</h1>', unsafe_allow_html=True)


    st.write("### Feature Important")
    # Mengambil fitur penting dari model yang sudah dilatih
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        "Fitur": X_test.columns,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    # Membuat visualisasi dengan grafik batang yang lebih profesional
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(feature_importance_df))
    sns.barplot(
        x="Importance",
        y="Fitur",
        data=feature_importance_df,
        palette=colors
    )
    plt.title("Feature Importance", fontsize=16, fontweight='bold')
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Fitur", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    

    # Menambahkan angka di atas setiap batang
    for index, value in enumerate(feature_importance_df["Importance"]):
        plt.text(
            value + 0.01,  # Posisi x (sedikit di sebelah kanan batang)
            index,         # Posisi y (sesuai indeks batang)
            f"{value:.2f}", # Nilai yang ditampilkan (dengan 2 desimal)
            va='center',   # Vertical alignment
            fontsize=10    # Ukuran font angka
        )

    plt.tight_layout()

    st.pyplot(plt)

    st.write("### Perbandingan Prediksi vs Aktual")
    comparison_df = pd.DataFrame({
        'Aktual': le.inverse_transform(y_test),
        'Prediksi': le.inverse_transform(pipeline.predict(X_test))
    })

    comparison_df['Hasil'] = np.where(comparison_df['Aktual'] == comparison_df['Prediksi'], 'Benar', 'Salah')

    # Membuat visualisasi pie chart untuk hasil prediksi
    fig = px.pie(
        comparison_df,
        names='Hasil',
        title="Distribusi Hasil Prediksi",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig)

    # Menampilkan hasil prediksi yang benar
    correct_predictions = (y_test == pipeline.predict(X_test)).sum()
    total_predictions = len(y_test)
    st.write(f"**Prediksi Benar:** {correct_predictions} dari {total_predictions} data")



def use_model_for_prediction():
    st.markdown('<h1 style="text-align: center; color: #1f4287;">🚀 Use Model for Predict</h1>', unsafe_allow_html=True)

    # Memastikan 'prediction_variable' diinisialisasi
    if 'prediction_variable' not in st.session_state:
        st.session_state.prediction_variable = []  # Atur default sebagai list kosong atau nilai yang sesuai

    load_model = st.checkbox("Muat model yang sudah disimpan untuk prediksi data baru")

    if load_model:
        progress_bar.progress(100)
        try:
            # Memuat model yang telah disimpan
            loaded_model = joblib.load('model_pass_fail.pkl')
            st.success("Model telah dimuat.")

            # Mengupload file CSV untuk data baru
            uploaded_file_new = st.file_uploader("Upload file CSV data baru", type=["csv"])

            if uploaded_file_new is not None:
                data_baru = pd.read_csv(uploaded_file_new)

                # Memastikan kolom prediktor sesuai dengan data pelatihan
                prediction_variable = st.session_state.prediction_variable
                if not prediction_variable:
                    st.error("Kolom prediktor belum diinisialisasi. Pastikan model telah dilatih sebelumnya.")
                    return

                if set(prediction_variable).issubset(data_baru.columns):
                    X_baru = data_baru[prediction_variable]
                    X_baru = pd.get_dummies(X_baru, drop_first=True)

                    # Menggunakan scaler yang ada di pipeline untuk standarisasi
                    scaler = loaded_model.named_steps['scaler']
                    X_baru_scaled = scaler.transform(X_baru)

                    # Melakukan prediksi
                    y_pred_baru = loaded_model.predict(X_baru_scaled)

                    # Menggunakan label encoder untuk inverse transform jika target bersifat kategorikal
                    le = st.session_state.le
                    y_pred_baru_labels = le.inverse_transform(y_pred_baru)

                    # Menambahkan kolom prediksi ke data baru
                    data_baru['Prediksi'] = y_pred_baru_labels

                    # Menghitung jumlah prediksi untuk setiap kategori
                    pred_count = data_baru['Prediksi'].value_counts()
                    labels = pred_count.index.tolist()
                    values = pred_count.values.tolist()

                    # Menghitung jumlah yang lulus dan tidak lulus
                    pass_count = np.sum(y_pred_baru_labels == 'Lulus')  # Ganti dengan label sesuai dengan dataset
                    fail_count = np.sum(y_pred_baru_labels == 'Tidak Lulus')  # Ganti dengan label sesuai dataset

                    # Menampilkan Pie Chart hasil prediksi
                    st.subheader("Hasil Prediksi")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie([pass_count, fail_count], labels=['Lulus', 'Tidak Lulus'], autopct='%1.1f%%', startangle=90, colors=['#28A745', '#B80000'])
                    ax.axis('equal')  # Memastikan pie chart berbentuk lingkaran
                    st.pyplot(fig)

                    # Menampilkan keterangan jumlah yang lulus dan tidak lulus
                    st.write(f"Jumlah yang Lulus: {pass_count} siswa")
                    st.write(f"Jumlah yang Tidak Lulus: {fail_count} siswa")
            

                    # Menampilkan preview data lengkap dengan kolom prediksi
                    st.subheader("Preview Data Lengkap dengan Hasil Prediksi")
                    st.dataframe(data_baru)
                else:
                    st.error("Kolom pada data baru tidak sesuai dengan yang digunakan pada model.")
            else:
                st.warning("Silakan unggah file CSV terlebih dahulu.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat model atau memproses data: {e}")



def main():
    # List halaman yang tersedia
    pages = ["Home","Input Dataset", "Preprocessing Data", "Analisis Data", "Data Visualization", "Use Model"]
    
    # Inisialisasi session state untuk navigasi
    if 'current_page_idx' not in st.session_state:
        st.session_state.current_page_idx = 0
        
    # Fungsi callback untuk tombol
    def next_page():
        if st.session_state.current_page_idx < len(pages) - 1:
            st.session_state.current_page_idx += 1
        
    def prev_page():
        if st.session_state.current_page_idx > 0:
            st.session_state.current_page_idx -= 1
    
    
    # Menambahkan menu navigasi di sidebar
    st.sidebar.selectbox(
        "Pilih Halaman", 
        pages, 
        key='navigation',
        index=st.session_state.current_page_idx
    )
    
    # Menambahkan teks di bagian bawah sidebar dengan posisi yang benar
    st.sidebar.markdown(
        """
        <div style="position: fixed; bottom: 30px;left: 45px; width: 100%; font-size: 20px; color: #555; text-align: left;">
            Made with ❤️ by Your Name
        </div>
        """, 
        unsafe_allow_html=True
    )  

    # Update current_page_idx berdasarkan selectbox
    st.session_state.current_page_idx = pages.index(st.session_state.navigation)
    
    # Menampilkan konten halaman yang dipilih
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
    elif current_page == "Use Model":
        use_model_for_prediction()
    
    # Membuat container untuk tombol navigasi
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_page_idx > 0:
            st.button('⬅️ Previous', on_click=prev_page, key='prev')
    
    with col3:
        if st.session_state.current_page_idx < len(pages) - 1:
            st.button('Next ➡️', on_click=next_page, key='next')

# Menjalankan aplikasi utama
if __name__ == "__main__":
    main()
