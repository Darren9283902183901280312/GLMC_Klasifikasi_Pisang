"""
app.py - Aplikasi klasifikasi citra kematangan pisang (ekstraksi fitur GLCM + machine learning).
Jalankan:  streamlit run app.py
Model dihasilkan oleh dataset_citra.ipynb dan disimpan di folder models/.
"""
import base64
import html
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import hog
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError: 
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

JUDUL = "Klasifikasi kematangan pisang"
DESKRIPSI = ("Unggah foto pisang. Aplikasi mengekstrak fitur tekstur (GLCM), warna, dan bentuk, "
             "lalu memprediksi tingkat kematangannya dengan machine learning.")
NAMA_DATASET = "Banana Ripeness Classification (Roboflow Universe), 6 kelas, format folder train/valid/test"
CATATAN_INPUT = "Gunakan foto pisang dengan latar sederhana dan pencahayaan cukup."
NAMA_TAMPIL = {
    "freshripe": "Fresh ripe (matang segar)",
    "freshunripe": "Fresh unripe (mentah segar)",
    "overripe": "Overripe (terlalu matang)",
    "ripe": "Ripe (matang)",
    "rotten": "Rotten (busuk)",
    "unripe": "Unripe (mentah)",
}
AMBANG_RENDAH = 0.50 

VERSI_FITUR = "glcm-v1"
UKURAN = 128 
LEVEL = 32 
JARAK = (1, 3) 
SUDUT = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
NAMA_SUDUT = (0, 45, 90, 135)
PROPS = ("contrast", "correlation", "ASM", "homogeneity")
NAMA_PROP = {"ASM": "energy"}


def baca_citra(path):
    """Baca gambar (aman untuk path yang mengandung spasi/karakter non-ASCII)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {path}")
    return img


def praproses(img_bgr, ukuran=UKURAN):
    """Resize lalu ubah ke grayscale. Mengembalikan (bgr, gray)."""
    bgr = cv2.resize(img_bgr, (ukuran, ukuran), interpolation=cv2.INTER_AREA)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def kuantisasi(gray, level=LEVEL):
    return (gray.astype(np.uint16) * level // 256).astype(np.uint8)


def hitung_glcm(gray, level=LEVEL, jarak=JARAK, sudut=SUDUT, symmetric=True):
    return graycomatrix(kuantisasi(gray, level), list(jarak), list(sudut),
                        levels=level, symmetric=symmetric, normed=True)


def homogeneity_kuliah(glcm):
    """sum p(i,j) / (1 + |i-j|). Bentuk (n_jarak, n_sudut)."""
    n = glcm.shape[0]
    i, j = np.indices((n, n))
    return np.einsum("ij,ijkl->kl", 1.0 / (1.0 + np.abs(i - j)), glcm)


def hitung_prop(glcm, p):
    return homogeneity_kuliah(glcm) if p == "homogeneity" else graycoprops(glcm, p)


def fitur_glcm(gray):
    glcm = hitung_glcm(gray)
    nilai, nama = [], []
    for p in PROPS:
        m = hitung_prop(glcm, p)
        for i, d in enumerate(JARAK):
            for j, a in enumerate(NAMA_SUDUT):
                nilai.append(m[i, j])
                nama.append(f"glcm_{NAMA_PROP.get(p, p)}_d{d}_a{a}")
    return np.nan_to_num(np.array(nilai, dtype=float)), nama


def fitur_warna(bgr):
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    nilai, nama = [], []
    for kanal, bins, rentang, label in ((h, 12, [0, 180], "h"), (s, 4, [0, 256], "s"), (v, 4, [0, 256], "v")):
        hist = cv2.calcHist([kanal], [0], None, [bins], rentang).ravel()
        hist = hist / (hist.sum() + 1e-9)
        for i, x in enumerate(hist):
            nilai.append(float(x))
            nama.append(f"warna_{label}{i}")
    merah = (((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)).mean()
    nilai += [float(merah), float((v < 70).mean())]
    nama += ["warna_frac_merah", "warna_frac_gelap"]
    return np.array(nilai, dtype=float), nama


def fitur_bentuk(gray):
    kecil = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    h = hog(kecil, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hu = cv2.HuMoments(cv2.moments(bw, binaryImage=True)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    nama = [f"bentuk_hog{i}" for i in range(len(h))] + [f"bentuk_hu{i}" for i in range(7)]
    return np.nan_to_num(np.concatenate([h, hu]).astype(float)), nama


def ekstrak_semua(img_bgr):
    """Praproses + semua fitur. Mengembalikan (vektor, nama_fitur, bgr, gray)."""
    bgr, gray = praproses(img_bgr)
    bagian = [fitur_glcm(gray), fitur_warna(bgr), fitur_bentuk(gray)]
    return (np.concatenate([b[0] for b in bagian]), sum([b[1] for b in bagian], []), bgr, gray)


def pilih_indeks(nama_fitur, kelompok):
    """Indeks kolom untuk kelompok fitur, mis. ('glcm', 'warna')."""
    return [i for i, n in enumerate(nama_fitur) if n.split("_")[0] in kelompok]


ROOT = Path(__file__).parent
DIR_MODELS, DIR_DATASET = ROOT / "models", ROOT / "Dataset"
ALGO_FILE = {"Naive Bayes": "naive_bayes", "Decision Tree": "decision_tree", "SVM": "svm", "KNN": "knn"}

st.set_page_config(page_title=JUDUL, page_icon="🍌", layout="wide")

# ---------------------------------------------------------------------------
# Tema pisang (otomatis mengikuti mode terang / gelap)
# ---------------------------------------------------------------------------
PALET = {
    "light": dict(panel="#FFFDF4", card="#FFFFFF", side="#FFF6D6", line="#E6DAA8", text="#2B2412",
                  muted="#6B6248", accent="#F2B705", track="#F3EBC8", warn="#C26A1B", ok="#2E7D32",
                  bad="#B3261E", cm=("#FFFBE0", "#F2B705", "#6B4400")),
    "dark": dict(panel="#1E1B12", card="#29251A", side="#26221A", line="#403A24", text="#F4EFD8",
                 muted="#B5AE92", accent="#FFD21F", track="#3A3520", warn="#E9A25A", ok="#7BD88F",
                 bad="#FF8A80", cm=("#2A2618", "#C98F00", "#FFE066")),
}


def mode_tema():
    """'light' / 'dark' dari Streamlit (>= 1.46); None bila tidak diketahui (ikut pengaturan sistem)."""
    try:
        t = st.context.theme.type
        return t if t in PALET else None
    except Exception:
        return None


MODE = mode_tema()
P = PALET[MODE or "light"]


@st.cache_resource
def pola_latar(nama_berkas):
    p = ROOT / "assets" / nama_berkas
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def vars_css(p):
    return (f"--panel:{p['panel']};--card:{p['card']};--side:{p['side']};--line:{p['line']};--text:{p['text']};"
            f"--muted:{p['muted']};--accent:{p['accent']};--track:{p['track']};--warn:{p['warn']};"
            f"--ok:{p['ok']};--bad:{p['bad']};")


def css_latar(p, berkas, overlay):
    b64 = pola_latar(berkas)
    gambar = f"url(data:image/jpeg;base64,{b64})" if b64 else "none"
    lapisan = f"linear-gradient({overlay},{overlay}),{gambar}" if overlay else gambar
    return f"--latar:{lapisan};"


_LT = f"{vars_css(PALET['light'])}{css_latar(PALET['light'], 'banana_pattern.jpg', 'rgba(255,253,244,.50)')}"
_DK = f"{vars_css(PALET['dark'])}{css_latar(PALET['dark'], 'banana_pattern_dark.jpg', '')}"
# Sumber kebenaran = tema yang benar-benar tampil di browser (skrip di bawah menulis data-tema).
# Hasil st.context.theme hanya dipakai sebagai petunjuk awal dan selalu dikalahkan oleh data-tema.
VARS = f":root{{{_LT}}}:root[data-tema=\"dark\"]{{{_DK}}}"
if MODE == "dark":
    VARS += f":root:not([data-tema=\"light\"]){{{_DK}}}"

# Penyesuaian khusus mode terang. Mode gelap tidak disentuh sama sekali.
# Teks & widget Streamlit dipaksa mengikuti palet pisang agar serasi dengan panel krem.
_L = ':root[data-tema="light"] ' if MODE == "dark" else ':root:not([data-tema="dark"]) '
if True:
    _teks = (".block-container, .block-container p, .block-container li, .block-container label, .block-container h1, "
             ".block-container h2, .block-container h3, .block-container h4, .block-container [data-testid=\"stMarkdownContainer\"], "
             "section[data-testid=\"stSidebar\"] p, section[data-testid=\"stSidebar\"] label, section[data-testid=\"stSidebar\"] h2, "
             "section[data-testid=\"stSidebar\"] [data-testid=\"stMarkdownContainer\"]")
    _muted = (".block-container [data-testid=\"stCaptionContainer\"], section[data-testid=\"stSidebar\"] "
              "[data-testid=\"stCaptionContainer\"], button[data-baseweb=\"tab\"]")
    _pre = lambda sel: ", ".join(_L + x.strip() for x in sel.split(", "))
    CSS_TERANG = (
        _pre(_teks) + " { color: var(--text); }\n"
        + _pre(_muted) + " { color: var(--muted); }\n"
        + _pre('button[data-baseweb="tab"][aria-selected="true"]') + " { color: var(--text); }\n"
        # kolom isian: latar putih + garis krem supaya terpisah dari panel
        + _pre('[data-testid="stFileUploaderDropzone"]') + " { background: var(--card); border: 1px solid var(--line); }\n"
        + _pre('div[data-baseweb="select"] > div') + " { background: var(--card); border-color: var(--line); color: var(--text); }\n"
        + _pre('[data-testid="stExpander"] details') + " { border-color: var(--line); background: var(--card); }\n"
        + _pre('[data-testid="stDataFrame"]') + " { border: 1px solid var(--line); border-radius: 10px; }\n"
    )

st.markdown("<style>\n@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');\n"
            + VARS + """
html, body, .stApp, [data-testid="stMarkdownContainer"] { font-family: 'Nunito', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Nunito', system-ui, sans-serif !important; font-weight: 800 !important; letter-spacing: -0.01em; }

/* Latar halaman: pola pisang. Konten ada di panel solid agar mudah dibaca. */
.stApp { background-image: var(--latar); background-size: 480px 480px; background-repeat: repeat; background-attachment: fixed; }
header[data-testid="stHeader"] { background: transparent; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1150px; background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    padding: 1.8rem 2.2rem 2.4rem 2.2rem !important; margin-top: 1rem; margin-bottom: 1.5rem; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: var(--side); }
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }

/* Hero: kuning pisang dengan teks gelap (kontras tinggi di kedua mode) */
.hero { background: linear-gradient(135deg, #FFDD3C 0%, #F7B801 100%); color: #2B2412;
        padding: 1.5rem 1.9rem; border-radius: 14px; margin-bottom: 1.3rem; }
.hero h1 { color: #2B2412; margin: 0 0 .35rem 0; font-size: 2rem; }
.hero p { margin: 0; color: #3A3016; max-width: 46rem; line-height: 1.55; font-weight: 600; }

/* Kartu hasil */
.hasil { border: 1px solid var(--line); border-left: 6px solid var(--accent); background: var(--card);
         padding: 1rem 1.3rem; border-radius: 12px; margin-bottom: 1rem; }
.hasil.rendah { border-left-color: var(--warn); }
.hasil .ket { color: var(--muted); font-size: .9rem; font-weight: 600; }
.hasil .kelas { font-weight: 800; font-size: 2rem; line-height: 1.15; margin: .15rem 0 .6rem 0;
                word-break: break-word; color: var(--text); }
.meter { background: var(--track); height: 10px; border-radius: 5px; overflow: hidden; }
.meter > div { background: var(--accent); height: 100%; border-radius: 5px; }
.hasil.rendah .meter > div { background: var(--warn); }
.hasil .nilai { margin-top: .45rem; font-size: .95rem; color: var(--text); }
.benar { color: var(--ok); font-weight: 700; } .salah { color: var(--bad); font-weight: 700; }

/* Tab, gambar */
button[data-baseweb="tab"] { font-weight: 700; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
[data-testid="stImage"] img { border-radius: 10px; }
""" + CSS_TERANG + "</style>", unsafe_allow_html=True)

try:
    import streamlit.components.v1 as components
    components.html("""<script>
    (function () {
      var p = window.parent.document;
      function lum(c) {
        var m = c && c.match(/[\\d.]+/g);
        if (!m || m.length < 3) return null;
        if (m.length > 3 && parseFloat(m[3]) === 0) return null;   // transparan
        return 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2];
      }
      function terapkan() {
        var app = p.querySelector('.stApp') || p.body;
        var l = lum(getComputedStyle(app).backgroundColor);        // latar tema Streamlit
        if (l === null) l = 255 - lum(getComputedStyle(app).color); // cadangan: warna teks
        var nilai = l < 128 ? 'dark' : 'light';
        if (p.documentElement.getAttribute('data-tema') !== nilai) p.documentElement.setAttribute('data-tema', nilai);
      }
      terapkan(); setInterval(terapkan, 250);
    })();
    </script>""", height=0)
    st.markdown("<style>[data-testid=\"stElementContainer\"]:has(iframe[height=\"0\"]){display:none;}</style>",
                unsafe_allow_html=True)
except Exception:
    pass


def gaya_matplotlib():
    """Samakan warna grafik matplotlib dengan panel agar terbaca di mode terang maupun gelap."""
    if MODE is None:  # tema baru diketahui di browser: pakai warna netral transparan
        plt.rcParams.update({
            "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
            "text.color": "#857F68", "axes.labelcolor": "#857F68", "axes.edgecolor": "#B5AE92",
            "xtick.color": "#857F68", "ytick.color": "#857F68"})
        return
    plt.rcParams.update({
        "figure.facecolor": P["panel"], "axes.facecolor": P["panel"], "savefig.facecolor": P["panel"],
        "text.color": P["text"], "axes.labelcolor": P["text"], "axes.edgecolor": P["line"],
        "xtick.color": P["text"], "ytick.color": P["text"],
    })


def cmap_pisang():
    return LinearSegmentedColormap.from_list("pisang", list(P["cm"]))


def warna_sel(rasio):
    """Warna angka di dalam sel confusion matrix, kontras terhadap warna selnya."""
    if MODE == "dark":
        return "#1E1B12" if rasio > 0.6 else "#F4EFD8"
    return "#FFFFFF" if rasio > 0.6 else "#2B2412"


gaya_matplotlib()

@st.cache_resource
def muat_semua():
    p = DIR_MODELS / "scaler.joblib"
    if not p.exists():
        return None, {}
    scaler = joblib.load(p)
    model = {n: joblib.load(DIR_MODELS / f"{f}.joblib") for n, f in ALGO_FILE.items()
             if (DIR_MODELS / f"{f}.joblib").exists()}
    return scaler, model


@st.cache_data
def daftar_contoh(per_kelas=4):
    """Contoh gambar dari Dataset/test (jika foldernya ada)."""
    for d in (DIR_DATASET.iterdir() if DIR_DATASET.exists() else []):
        if d.is_dir() and d.name.lower() in ("test", "testing"):
            hasil = []
            for k in sorted(p for p in d.iterdir() if p.is_dir()):
                files = sorted(f for f in k.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
                hasil += [(k.name, str(f)) for f in files[:per_kelas]]
            return hasil
    return []


def lebar_penuh(fn, *args, **kw):
    """Panggil elemen Streamlit dengan lebar penuh, kompatibel lintas versi:
    versi baru memakai width="stretch", versi lama use_container_width=True."""
    for ekstra in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            return fn(*args, **kw, **ekstra)
        except Exception:
            continue
    return fn(*args, **kw)


def nama(kelas):
    return NAMA_TAMPIL.get(str(kelas), str(kelas))


def prediksi(model, scaler, vektor, nama_fitur):
    """Daftar (kelas, probabilitas) terurut dari yang tertinggi."""
    posisi = {n: i for i, n in enumerate(nama_fitur)}
    idx = [posisi[n] for n in scaler.info_["nama_fitur"]]
    x = scaler.transform(vektor[idx].reshape(1, -1))
    proba = model.predict_proba(x)[0]
    urut = np.argsort(proba)[::-1]
    return [(str(model.classes_[i]), float(proba[i])) for i in urut]


def baca_unggahan(file):
    return cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)


scaler, models = muat_semua()

st.markdown(f'<div class="hero"><h1> {html.escape(JUDUL)}</h1><p>{html.escape(DESKRIPSI)}</p></div>',
            unsafe_allow_html=True)

if scaler is None or not models:
    st.error("Model belum tersedia di folder models/. Jalankan dataset_citra.ipynb sampai selesai "
             "(sel terakhir menyimpan model), lalu buka ulang aplikasi ini.")
    st.stop()
if scaler.info_.get("versi_fitur") != VERSI_FITUR:
    st.error("Versi ekstraksi fitur pada model berbeda dengan app.py. Jalankan ulang notebook untuk melatih ulang model.")
    st.stop()

nama_terbaik = next((n for n, m in models.items() if m.info_.get("terbaik")), list(models)[0])

with st.sidebar:
    st.header("Masukan gambar")
    sumber = st.radio("Sumber gambar", ["Unggah gambar", "Pilih contoh"], label_visibility="collapsed")
    gambar_bgr, nama_gambar, kelas_asli = None, "", None
    if sumber == "Unggah gambar":
        f = st.file_uploader("Foto (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if f is not None:
            try:
                gambar_bgr, nama_gambar = baca_unggahan(f), f.name
            except Exception:
                st.error("File tidak dapat dibaca sebagai gambar. Coba file JPG atau PNG lain.")
    else:
        contoh = daftar_contoh()
        if contoh:
            pilih = st.selectbox("Contoh dari data test", contoh,
                                 format_func=lambda c: f"{nama(c[0])} - {Path(c[1]).name[:18]}")
            gambar_bgr, nama_gambar, kelas_asli = baca_citra(pilih[1]), Path(pilih[1]).name, pilih[0]
        else:
            st.info("Folder Dataset/test tidak ditemukan, jadi contoh gambar tidak tersedia. Unggah gambar sendiri.")

    st.divider()
    algo = st.selectbox("Algoritma", list(models), index=list(models).index(nama_terbaik),
                        format_func=lambda n: n + (" (terbaik)" if n == nama_terbaik else ""))
    model = models[algo]
    st.caption(f"Fitur: {scaler.info_['set_fitur']}")
    st.caption(CATATAN_INPUT)

tab_hasil, tab_proses, tab_model, tab_tentang = st.tabs(
    ["Hasil klasifikasi", "Proses ekstraksi fitur", "Performa model", "Tentang"])

if gambar_bgr is None:
    with tab_hasil:
        st.info("Mulai dengan mengunggah foto atau memilih contoh gambar di panel kiri.")
    with tab_proses:
        st.info("Tahap pemrosesan akan tampil setelah ada gambar.")
else:
    vektor, nama_fitur, bgr, gray = ekstrak_semua(gambar_bgr)
    hasil = prediksi(model, scaler, vektor, nama_fitur)
    top, p_top = hasil[0]
    rendah = p_top < AMBANG_RENDAH

    with tab_hasil:
        kiri, kanan = st.columns([1, 1], gap="large")
        with kiri:
            lebar_penuh(st.image, cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB), caption=nama_gambar)
        with kanan:
            st.markdown(
                f'<div class="hasil {"rendah" if rendah else ""}">'
                f'<div class="ket">Hasil prediksi ({html.escape(algo)})</div>'
                f'<div class="kelas">{html.escape(nama(top))}</div>'
                f'<div class="meter"><div style="width:{p_top * 100:.0f}%"></div></div>'
                f'<div class="nilai">Peluang {p_top:.1%}</div></div>', unsafe_allow_html=True)
            if kelas_asli is not None:
                cocok = kelas_asli == top
                st.markdown(f'Kelas sebenarnya: **{html.escape(nama(kelas_asli))}** '
                            f'<span class="{"benar" if cocok else "salah"}">{"(prediksi benar)" if cocok else "(prediksi salah)"}</span>',
                            unsafe_allow_html=True)
            if rendah:
                st.warning("Keyakinan model rendah. Coba foto yang lebih jelas atau lebih mirip data latih.")
            st.caption("Peluang tiap kelas")
            df_p = pd.DataFrame({"Kelas": [nama(k) for k, _ in hasil], "Peluang": [p for _, p in hasil]})
            lebar_penuh(st.dataframe, df_p, hide_index=True,
                        column_config={"Peluang": st.column_config.ProgressColumn(
                            "Peluang", min_value=0.0, max_value=1.0, format="%.2f")})

        st.subheader("Perbandingan keempat algoritma")
        baris = []
        for n, m in models.items():
            k, p = prediksi(m, scaler, vektor, nama_fitur)[0]
            baris.append({"Algoritma": n + (" (terbaik)" if n == nama_terbaik else ""),
                          "Prediksi": nama(k), "Peluang": round(p, 3)})
        lebar_penuh(st.dataframe, pd.DataFrame(baris), hide_index=True)

    with tab_proses:
        info = scaler.info_
        st.write(f"Gambar diubah ke ukuran {UKURAN}×{UKURAN} dan grayscale, tingkat keabuannya diturunkan "
                 f"dari 256 menjadi {LEVEL}, lalu matriks GLCM dihitung untuk jarak "
                 f"{', '.join(map(str, JARAK))} dan sudut 0°, 45°, 90°, 135°.")
        a, b, c = st.columns(3)
        lebar_penuh(a.image, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Setelah resize")
        lebar_penuh(b.image, gray, caption="Grayscale")
        q = kuantisasi(gray)
        lebar_penuh(c.image, (q.astype(float) * 255 / (LEVEL - 1)).astype(np.uint8), caption=f"Kuantisasi {LEVEL} level")

        glcm = hitung_glcm(gray)
        rata = glcm[:, :, 0, :].mean(axis=2) 
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(np.log1p(rata * 1e3), cmap=cmap_pisang())
        ax.set_xlabel("Level keabuan j")
        ax.set_ylabel("Level keabuan i")
        ax.set_title("Matriks GLCM (jarak 1, rata-rata 4 sudut)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + 1000·p)")
        plt.tight_layout()
        g1, g2 = st.columns([1, 1.3], gap="large")
        g1.pyplot(fig)
        plt.close(fig)

        baris = []
        for p in PROPS:
            m = hitung_prop(glcm, p)
            for i, d in enumerate(JARAK):
                baris.append({"Fitur": NAMA_PROP.get(p, p).capitalize(), "Jarak": d,
                              **{f"{s}°": round(float(m[i, j]), 4) for j, s in enumerate(NAMA_SUDUT)}})
        g2.caption("Nilai fitur GLCM")
        lebar_penuh(g2.dataframe, pd.DataFrame(baris), hide_index=True)
        n_kel = {k: sum(n.startswith(k + "_") for n in nama_fitur) for k in ("glcm", "warna", "bentuk")}
        st.caption(f"Seluruh fitur yang diekstraksi: {len(nama_fitur)} (GLCM {n_kel['glcm']}, warna {n_kel['warna']}, "
                   f"bentuk {n_kel['bentuk']}). Model memakai set fitur: {info['set_fitur']} "
                   f"({len(info['nama_fitur'])} fitur).")

with tab_model:
    st.subheader("Hasil evaluasi pada data test")
    df = pd.DataFrame([{
        "Algoritma": n, "Parameter": str(m.info_["parameter"]), "F1 valid": m.info_["f1_valid"],
        "F1 k-fold": m.info_["f1_cv"], "Precision": m.info_["precision"], "Recall": m.info_["recall"],
        "F1-score": m.info_["f1"], "Akurasi": m.info_["akurasi"],
        "Waktu latih (s)": m.info_["waktu_latih"], "Waktu prediksi (s)": m.info_["waktu_prediksi"]}
        for n, m in models.items()])
    lebar_penuh(st.dataframe, df.round(4), hide_index=True)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.caption("Perbandingan metrik pada data test")
        st.bar_chart(df.set_index("Algoritma")[["Precision", "Recall", "F1-score", "Akurasi"]],
                     color=["#F2B705", "#8B5A00", "#6B8E23", "#C26A1B"])
    with c2:
        st.caption(f"Confusion matrix: {algo}")
        kelas = model.info_["kelas"]
        cm = np.array(model.info_["confusion"])
        fig, ax = plt.subplots(figsize=(5, 4.4))
        besar = len(kelas) > 12
        ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(min=1) if besar else cm, cmap=cmap_pisang())
        ax.set_xticks(range(len(kelas)))
        ax.set_yticks(range(len(kelas)))
        ax.set_xticklabels(kelas, rotation=45 if not besar else 90, ha="right" if not besar else "center",
                           fontsize=8 if not besar else 5)
        ax.set_yticklabels(kelas, fontsize=8 if not besar else 5)
        if not besar:
            for i in range(len(kelas)):
                for j in range(len(kelas)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color=warna_sel(cm[i, j] / max(cm.max(), 1)))
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Sebenarnya")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    info = scaler.info_
    st.caption(f"Model terbaik dipilih dari F1 pada data valid. Parameter dituning pada data valid, "
               f"k-fold ({info['k_fold']} fold) dihitung pada data train. Waktu ekstraksi fitur rata-rata "
               f"{info['waktu_ekstraksi_rata2_ms']:.1f} ms per citra.")

with tab_tentang:
    st.markdown(f"""
**Dataset.** {NAMA_DATASET}.

**Metode.** Fitur tekstur GLCM (Contrast, Correlation, Energy, Homogeneity) dapat digabung dengan
histogram warna HSV dan fitur bentuk (HOG, Hu Moments). Set fitur dipilih berdasarkan F1-score pada
data valid. Fitur dinormalisasi dengan StandardScaler lalu diklasifikasikan dengan Naive Bayes,
Decision Tree, SVM, dan KNN.
""")
    st.write("Jumlah citra per kelas:")
    ps = scaler.info_["jumlah_per_split"]
    dfk = pd.DataFrame(ps).fillna(0).astype(int)
    dfk.index = [nama(k) for k in dfk.index]
    st.dataframe(dfk)
