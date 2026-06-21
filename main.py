"""
=============================================================================
ANALISIS PREDIKSI HARGA SAHAM ANTM (Aneka Tambang)
Bursa Efek Indonesia
=============================================================================
Model: XGBoost & LSTM
Target: Prediksi Harga Close
Evaluasi: MAE, RMSE, MAPE, R²

Dataset: https://www.kaggle.com/datasets/tiwill/saham-idx
=============================================================================
"""

# =============================================================================
# IMPORT LIBRARY
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATASET_DIR, "ANTM.csv")

# Hyperparameter
WINDOW_SIZE   = 20    # Jumlah hari untuk sliding window LSTM
TEST_SIZE     = 0.20  # Rasio data uji
LSTM_EPOCHS   = 100
LSTM_BATCH    = 32
RANDOM_STATE  = 42

# Warna tema plotting
COLORS = {
    "actual"     : "#4FC3F7",
    "xgboost"    : "#FF7043",
    "lstm"       : "#66BB6A",
    "highlight"  : "#FDD835",
    "bg"         : "#1A1A2E",
    "grid"       : "#2E2E4E",
}

plt.rcParams.update({
    "figure.facecolor"  : COLORS["bg"],
    "axes.facecolor"    : COLORS["bg"],
    "axes.edgecolor"    : "#444466",
    "axes.labelcolor"   : "white",
    "xtick.color"       : "white",
    "ytick.color"       : "white",
    "text.color"        : "white",
    "grid.color"        : COLORS["grid"],
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "font.family"       : "DejaVu Sans",
    "legend.facecolor"  : "#1A1A2E",
    "legend.edgecolor"  : "#444466",
})


# =============================================================================
# 1. LOAD & EDA (EXPLORATORY DATA ANALYSIS)
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    print("\n" + "="*70)
    print("  [1/6] LOAD DATASET")
    print("="*70)

    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  • File    : {path}")
    print(f"  • Shape   : {df.shape[0]} baris × {df.shape[1]} kolom")
    print(f"  • Rentang : {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """EDA: info, statistik deskriptif, missing values, distribusi, korelasi."""
    print("\n" + "="*70)
    print("  [2/6] EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    # --- Info Umum ---
    print("\n[INFO KOLOM]")
    print(df.dtypes.to_string())

    print("\n[STATISTIK DESKRIPTIF]")
    print(df.describe().T.to_string())

    print("\n[MISSING VALUES]")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Jumlah": missing, "Persen (%)": missing_pct})
    missing_df = missing_df[missing_df["Jumlah"] > 0]
    if missing_df.empty:
        print("  Tidak ada missing value pada kolom yang digunakan.")
    else:
        print(missing_df.to_string())

    # --- Plot 1: Harga Close Historis ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    fig.suptitle("ANTM — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

    # Harga Close
    ax1 = axes[0]
    ax1.plot(df["date"], df["close"], color=COLORS["actual"], linewidth=1.5, label="Close Price")
    ax1.fill_between(df["date"], df["low"], df["high"], alpha=0.15, color=COLORS["actual"], label="Low–High Range")
    ax1.set_title("Harga Saham ANTM (Close, High, Low)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Harga (Rp)")
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Volume
    ax2 = axes[1]
    ax2.bar(df["date"], df["volume"], color=COLORS["lstm"], alpha=0.7, width=1.5, label="Volume")
    ax2.set_title("Volume Transaksi Harian", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Volume (Lot)")
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Change (perubahan harga harian)
    ax3 = axes[2]
    colors_change = [COLORS["lstm"] if c >= 0 else COLORS["xgboost"] for c in df["change"]]
    ax3.bar(df["date"], df["change"], color=colors_change, alpha=0.8, width=1.5, label="Daily Change")
    ax3.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax3.set_title("Perubahan Harga Harian (Change)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Perubahan (Rp)")
    ax3.legend()
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "01_eda_timeseries.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Simpan: {out_path}")

    # --- Plot 2: Distribusi & Korelasi ---
    fitur_utama = ["open_price", "high", "low", "close", "volume", "value",
                   "frequency", "foreign_buy", "foreign_sell", "change"]
    df_fitur = df[fitur_utama].dropna()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Distribusi Variabel Utama", fontsize=14, fontweight="bold")
    for i, col in enumerate(fitur_utama):
        r, c = divmod(i, 5)
        axes[r, c].hist(df_fitur[col], bins=40, color=COLORS["actual"], edgecolor="black", alpha=0.8)
        axes[r, c].set_title(col, fontsize=10)
        axes[r, c].set_xlabel("Nilai")
        axes[r, c].set_ylabel("Frekuensi")
        axes[r, c].grid(True)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_eda_distribusi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    # --- Plot 3: Heatmap Korelasi ---
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df_fitur.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9})
    ax.set_title("Heatmap Korelasi Antar Variabel", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_eda_korelasi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")


# =============================================================================
# 2. PREPROCESSING DATA
# =============================================================================

def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Praproses:
    - Hapus kolom tidak relevan
    - Tangani missing value
    - Deteksi & tangani outlier (IQR)
    - Feature engineering: net_foreign, MA, lag features, RSI, volatility
    """
    print("\n" + "="*70)
    print("  [3/6] PREPROCESSING DATA")
    print("="*70)

    df = df.copy()

    # ---- Pilih kolom yang relevan ----
    kolom_pakai = ["date", "open_price", "high", "low", "close",
                   "volume", "value", "frequency", "foreign_buy", "foreign_sell", "change"]
    df = df[kolom_pakai].copy()

    # ---- Tangani nilai 0 yang mencurigakan di open_price ----
    # Pada periode Maret–April 2020, open_price = 0 (data BEI tidak mencatat)
    # Ganti dengan previous (forward-fill setelah backward-fill)
    df["open_price"] = df["open_price"].replace(0, np.nan)
    df["open_price"] = df["open_price"].bfill().ffill()

    # ---- Tangani missing value: ffill + bfill ----
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    n_missing_after = df.isnull().sum().sum()
    print(f"  • Missing value setelah penanganan : {n_missing_after}")

    # ---- Deteksi Outlier dengan IQR pada kolom numerik ----
    # PERBAIKAN: Q1/Q3 dihitung HANYA dari porsi data yang akan jadi data latih
    # (perkiraan berdasarkan TEST_SIZE), bukan dari seluruh df. Jika dihitung dari
    # seluruh df (train+test), batas clipping ikut "melihat" distribusi periode
    # uji -> kebocoran informasi (data leakage) yang sifatnya halus namun nyata.
    numerik = ["close", "volume", "value", "frequency", "foreign_buy", "foreign_sell"]
    n_fit = int(len(df) * (1 - TEST_SIZE))  # batas approx data latih (kronologis)
    df_fit_outlier = df.iloc[:n_fit]
    for col in numerik:
        Q1 = df_fit_outlier[col].quantile(0.25)
        Q3 = df_fit_outlier[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        # Winsorize (clip) outlier ekstrem -- bounds dari train, diterapkan ke semua
        df[col] = df[col].clip(lower, upper)
        print(f"  • Outlier [{col:15s}] : {outlier_count:5d} → di-clip ke [{lower:,.0f}, {upper:,.0f}] (bounds dari data latih saja)")

    # ---- Feature Engineering ----
    # Net foreign flow
    df["net_foreign"] = df["foreign_buy"] - df["foreign_sell"]

    # Moving Averages
    for w in [5, 10, 20, 50]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()

    # Exponential Moving Average
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]

    # Bollinger Bands (20-hari)
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # RSI (14-hari)
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volatilitas (std 5-hari dari return harian)
    df["daily_return"] = df["close"].pct_change()
    df["volatility_5"] = df["daily_return"].rolling(5).std()

    # Lag Features (1, 3, 5 hari)
    for lag in [1, 3, 5]:
        df[f"close_lag{lag}"]  = df["close"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)

    # Target: close esok hari (prediksi 1-hari ke depan)
    df["target"] = df["close"].shift(-1)

    # Hapus baris dengan NaN setelah feature engineering
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n  • Shape setelah preprocessing : {df.shape[0]} baris × {df.shape[1]} kolom")
    print(f"  • Fitur engineering ditambahkan:")
    fitur_baru = ["net_foreign", "ma_5", "ma_10", "ma_20", "ma_50",
                  "ema_12", "ema_26", "macd", "bb_mid", "bb_upper", "bb_lower",
                  "bb_width", "rsi", "daily_return", "volatility_5",
                  "close_lag1", "close_lag3", "close_lag5",
                  "volume_lag1", "volume_lag3", "volume_lag5", "target"]
    for f in fitur_baru:
        print(f"    - {f}")

    return df


# =============================================================================
# 3. PEMBAGIAN DATA & NORMALISASI
# =============================================================================

def split_and_scale(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    Pembagian data 80/20 secara berurutan (time-series split).
    Normalisasi MinMax untuk LSTM.
    """
    print("\n" + "="*70)
    print("  PEMBAGIAN DATA & NORMALISASI")
    print("="*70)

    fitur_kolom = [c for c in df.columns if c not in ["date", "target"]]
    target_kolom = "target"

    X = df[fitur_kolom].values
    y = df[target_kolom].values

    split_idx = int(len(df) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scaler untuk fitur (untuk LSTM)
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)

    # Scaler untuk target (untuk inverse transform prediksi LSTM)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"  • Total data       : {len(df)}")
    print(f"  • Data latih       : {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"  • Data uji         : {len(X_test)} ({test_size*100:.0f}%)")
    print(f"  • Jumlah fitur     : {X_train.shape[1]}")
    print(f"  • Tanggal split    : {df['date'].iloc[split_idx].date()}")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled, "y_test_scaled": y_test_scaled,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "fitur_kolom": fitur_kolom,
        "df_test": df.iloc[split_idx:].reset_index(drop=True),
    }


def build_lstm_sequences(X_scaled: np.ndarray, y_scaled: np.ndarray,
                          window: int = WINDOW_SIZE):
    """Membentuk sliding window untuk input LSTM."""
    Xs, ys = [], []
    for i in range(window, len(X_scaled)):
        Xs.append(X_scaled[i - window: i])
        ys.append(y_scaled[i])
    return np.array(Xs), np.array(ys)


# =============================================================================
# 4. MODEL XGBOOST
# =============================================================================

def bangun_model_xgboost(data: dict) -> tuple:
    """
    Membangun, melatih, dan memprediksi menggunakan XGBoost.
    Mengembalikan model dan prediksi pada data uji.
    """
    print("\n" + "="*70)
    print("  [4a/6] PEMBANGUNAN MODEL XGBOOST")
    print("="*70)

    model = xgb.XGBRegressor(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = RANDOM_STATE,
        verbosity        = 0,
        early_stopping_rounds = 20,
        eval_metric      = "rmse",
    )

    eval_set = [(data["X_train"], data["y_train"]),
                (data["X_test"],  data["y_test"])]

    model.fit(
        data["X_train"], data["y_train"],
        eval_set=eval_set,
        verbose=False,
    )

    best_iter = model.best_iteration
    y_pred_xgb = model.predict(data["X_test"])

    print(f"  • Best iteration : {best_iter}")
    print(f"  • Prediksi rentang : [{y_pred_xgb.min():,.2f}, {y_pred_xgb.max():,.2f}]")

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    importances = pd.Series(model.feature_importances_, index=data["fitur_kolom"])
    importances_sorted = importances.nlargest(20)
    importances_sorted.sort_values().plot(kind="barh", ax=ax,
        color=[COLORS["xgboost"] if v > importances_sorted.median() else COLORS["actual"]
               for v in importances_sorted.sort_values()])
    ax.set_title("XGBoost — Top 20 Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "04_xgb_feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    return model, y_pred_xgb


# =============================================================================
# 5. MODEL LSTM
# =============================================================================

def bangun_model_lstm(data: dict, window: int = WINDOW_SIZE) -> tuple:
    """
    Membangun, melatih, dan memprediksi menggunakan LSTM.
    Mengembalikan model, history pelatihan, dan prediksi pada data uji.
    """
    print("\n" + "="*70)
    print("  [4b/6] PEMBANGUNAN MODEL LSTM")
    print("="*70)

    # Gabungkan train + test scaled untuk membentuk sequences
    X_all_scaled = np.vstack([data["X_train_scaled"], data["X_test_scaled"]])
    y_all_scaled = np.concatenate([data["y_train_scaled"], data["y_test_scaled"]])

    X_seq_all, y_seq_all = build_lstm_sequences(X_all_scaled, y_all_scaled, window)

    n_train = len(data["X_train_scaled"]) - window
    X_seq_train = X_seq_all[:n_train]
    y_seq_train = y_seq_all[:n_train]
    X_seq_test  = X_seq_all[n_train:]
    y_seq_test  = y_seq_all[n_train:]

    n_features = X_seq_train.shape[2]
    print(f"  • Window size      : {window}")
    print(f"  • Shape train seq  : {X_seq_train.shape}")
    print(f"  • Shape test seq   : {X_seq_test.shape}")
    print(f"  • Jumlah fitur     : {n_features}")

    # PERBAIKAN ARSITEKTUR:
    # 1) BatchNormalization diganti LayerNormalization. BatchNorm menghitung
    #    statistik antar-CONTOH dalam satu batch, yang tidak cocok untuk data
    #    time series yang non-stasioner (statistik training-time bisa tidak
    #    relevan lagi saat rezim harga bergeser di periode uji). LayerNorm
    #    menormalisasi per-sample/per-timestep sehingga tidak bergantung pada
    #    statistik batch -> jauh lebih stabil untuk RNN/LSTM.
    # 2) Arsitektur disederhanakan (2 layer LSTM, unit lebih kecil) karena
    #    jumlah sequence latih (~900-an setelah windowing) relatif sedikit
    #    untuk model setumpuk 3 layer LSTM + banyak parameter -> rawan
    #    menangkap pola yang rapuh terhadap pergeseran distribusi.
    # 3) Gradient clipping (clipnorm) ditambahkan untuk mencegah lonjakan
    #    gradien yang bisa memicu prediksi "meledak" pada data uji.
    from tensorflow.keras.layers import LayerNormalization

    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(window, n_features)),
        LayerNormalization(),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        LayerNormalization(),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss="huber",
                  metrics=["mae"])

    print("\n[ARSITEKTUR LSTM]")
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_seq_train, y_seq_train,
        epochs          = LSTM_EPOCHS,
        batch_size      = LSTM_BATCH,
        validation_split= 0.15,
        callbacks       = callbacks,
        verbose         = 1,
        shuffle         = False,  # Time series: jangan diacak
    )

    # Prediksi (inverse transform)
    y_pred_scaled = model.predict(X_seq_test, verbose=0).flatten()
    y_pred_lstm   = data["scaler_y"].inverse_transform(
                        y_pred_scaled.reshape(-1, 1)).flatten()

    # Plot Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM — Training History", fontsize=13, fontweight="bold")

    axes[0].plot(history.history["loss"],     color=COLORS["lstm"],    label="Train Loss")
    axes[0].plot(history.history["val_loss"], color=COLORS["xgboost"], label="Val Loss")
    axes[0].set_title("Loss (Huber)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["mae"],     color=COLORS["lstm"],    label="Train MAE")
    axes[1].plot(history.history["val_mae"], color=COLORS["xgboost"], label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "05_lstm_training_history.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Simpan: {out_path}")

    return model, history, y_pred_lstm, len(X_seq_test)


# =============================================================================
# 6. EVALUASI MODEL
# =============================================================================

def hitung_metrik(y_true: np.ndarray, y_pred: np.ndarray, nama_model: str) -> dict:
    """Menghitung MAE, RMSE, MAPE, dan R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)

    print(f"\n  [{nama_model}]")
    print(f"    MAE  : {mae:>12,.4f}")
    print(f"    RMSE : {rmse:>12,.4f}")
    print(f"    MAPE : {mape:>12,.4f} %")
    print(f"    R²   : {r2:>12,.4f}")

    return {"Model": nama_model, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def evaluasi_model(y_test: np.ndarray, y_pred_xgb: np.ndarray,
                   y_pred_lstm: np.ndarray, n_lstm: int) -> pd.DataFrame:
    """Evaluasi dan bandingkan kedua model."""
    print("\n" + "="*70)
    print("  [5/6] EVALUASI MODEL")
    print("="*70)

    # LSTM hanya punya prediksi untuk n_lstm data terakhir (karena window)
    y_test_lstm = y_test[-n_lstm:]
    y_test_xgb  = y_test  # XGBoost tidak butuh window

    metrik_xgb  = hitung_metrik(y_test_xgb,  y_pred_xgb,  "XGBoost")
    metrik_lstm = hitung_metrik(y_test_lstm, y_pred_lstm, "LSTM")

    df_metrik = pd.DataFrame([metrik_xgb, metrik_lstm])
    df_metrik.set_index("Model", inplace=True)

    # Simpan ke CSV
    csv_path = os.path.join(OUTPUT_DIR, "evaluasi_metrik.csv")
    df_metrik.to_csv(csv_path)
    print(f"\n  ✓ Tabel metrik disimpan: {csv_path}")
    print("\n[TABEL PERBANDINGAN METRIK]")
    print(df_metrik.to_string())

    return df_metrik, y_test_xgb, y_test_lstm


# =============================================================================
# 7. VISUALISASI HASIL
# =============================================================================

def visualisasi_hasil(data: dict, y_pred_xgb: np.ndarray,
                      y_pred_lstm: np.ndarray, n_lstm: int,
                      df_metrik: pd.DataFrame,
                      y_test_xgb: np.ndarray,
                      y_test_lstm: np.ndarray) -> None:
    """Visualisasi lengkap hasil prediksi kedua model."""
    print("\n" + "="*70)
    print("  [6/6] VISUALISASI HASIL")
    print("="*70)

    df_test  = data["df_test"]
    dates    = df_test["date"].values
    dates_xgb  = dates
    dates_lstm = dates[-n_lstm:]

    # =========================================================================
    # Plot A: Prediksi vs Aktual — XGBoost
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates_xgb, y_test_xgb,  color=COLORS["actual"],  linewidth=1.5,
            label="Aktual",    alpha=0.9)
    ax.plot(dates_xgb, y_pred_xgb,  color=COLORS["xgboost"], linewidth=1.5,
            label="XGBoost",   alpha=0.9, linestyle="--")
    ax.set_title("XGBoost — Prediksi vs Aktual Harga Close ANTM",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp)")
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Anotasi metrik
    mae_xgb  = df_metrik.loc["XGBoost", "MAE"]
    rmse_xgb = df_metrik.loc["XGBoost", "RMSE"]
    r2_xgb   = df_metrik.loc["XGBoost", "R2"]
    mape_xgb = df_metrik.loc["XGBoost", "MAPE"]
    ax.text(0.01, 0.96,
            f"MAE={mae_xgb:,.1f}  RMSE={rmse_xgb:,.1f}  MAPE={mape_xgb:.2f}%  R²={r2_xgb:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2E2E4E", alpha=0.8))
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_prediksi_xgboost.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    # =========================================================================
    # Plot B: Prediksi vs Aktual — LSTM
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates_lstm, y_test_lstm,  color=COLORS["actual"], linewidth=1.5,
            label="Aktual",   alpha=0.9)
    ax.plot(dates_lstm, y_pred_lstm,  color=COLORS["lstm"],   linewidth=1.5,
            label="LSTM",     alpha=0.9, linestyle="--")
    ax.set_title("LSTM — Prediksi vs Aktual Harga Close ANTM",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp)")
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    mae_lstm  = df_metrik.loc["LSTM", "MAE"]
    rmse_lstm = df_metrik.loc["LSTM", "RMSE"]
    r2_lstm   = df_metrik.loc["LSTM", "R2"]
    mape_lstm = df_metrik.loc["LSTM", "MAPE"]
    ax.text(0.01, 0.96,
            f"MAE={mae_lstm:,.1f}  RMSE={rmse_lstm:,.1f}  MAPE={mape_lstm:.2f}%  R²={r2_lstm:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2E2E4E", alpha=0.8))
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "07_prediksi_lstm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    # =========================================================================
    # Plot C: Perbandingan kedua model (pada periode LSTM)
    # =========================================================================
    # Sejajarkan XGBoost pada periode yang sama dengan LSTM
    y_pred_xgb_overlap = y_pred_xgb[-n_lstm:]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(dates_lstm, y_test_lstm,        color=COLORS["actual"],  linewidth=2.0,
            label="Aktual",    alpha=1.0, zorder=3)
    ax.plot(dates_lstm, y_pred_xgb_overlap, color=COLORS["xgboost"], linewidth=1.5,
            label="XGBoost",   alpha=0.85, linestyle="--", zorder=2)
    ax.plot(dates_lstm, y_pred_lstm,        color=COLORS["lstm"],    linewidth=1.5,
            label="LSTM",      alpha=0.85, linestyle="-.", zorder=2)

    ax.fill_between(dates_lstm, y_pred_xgb_overlap, y_test_lstm,
                    alpha=0.15, color=COLORS["xgboost"])
    ax.fill_between(dates_lstm, y_pred_lstm, y_test_lstm,
                    alpha=0.15, color=COLORS["lstm"])

    ax.set_title("Perbandingan XGBoost vs LSTM — Harga Close ANTM",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp)")
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "08_perbandingan_model.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    # =========================================================================
    # Plot D: Bar Chart Perbandingan Metrik
    # =========================================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Perbandingan Metrik Evaluasi Model", fontsize=14, fontweight="bold")

    metrik_list = ["MAE", "RMSE", "MAPE", "R2"]
    labels      = ["MAE (Rp)", "RMSE (Rp)", "MAPE (%)", "R²"]

    for i, (met, label) in enumerate(zip(metrik_list, labels)):
        vals   = [df_metrik.loc["XGBoost", met], df_metrik.loc["LSTM", met]]
        colors = [COLORS["xgboost"], COLORS["lstm"]]
        bars   = axes[i].bar(["XGBoost", "LSTM"], vals, color=colors,
                              edgecolor="white", linewidth=0.8, width=0.5)
        axes[i].set_title(label, fontsize=12)
        axes[i].grid(True, axis="y")
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.02,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                         fontweight="bold")
        axes[i].set_ylim(0, max(vals) * 1.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "09_metrik_bar_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    # =========================================================================
    # Plot E: Scatter Plot Aktual vs Prediksi
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Scatter Plot: Aktual vs Prediksi", fontsize=14, fontweight="bold")

    for ax, y_true, y_pred, nama, color in [
        (axes[0], y_test_xgb,  y_pred_xgb,  "XGBoost", COLORS["xgboost"]),
        (axes[1], y_test_lstm, y_pred_lstm, "LSTM",    COLORS["lstm"]),
    ]:
        ax.scatter(y_true, y_pred, color=color, alpha=0.5, s=20, label=nama)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="white", linestyle="--",
                linewidth=1.5, label="Ideal (y=x)")
        ax.set_title(f"{nama}", fontsize=12)
        ax.set_xlabel("Aktual (Rp)")
        ax.set_ylabel("Prediksi (Rp)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "10_scatter_aktual_vs_prediksi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    # =========================================================================
    # Plot F: Residual Analysis
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analisis Residual", fontsize=14, fontweight="bold")

    resid_xgb  = y_test_xgb  - y_pred_xgb
    resid_lstm = y_test_lstm - y_pred_lstm

    # Residual time-plot
    axes[0, 0].plot(dates_xgb,  resid_xgb,  color=COLORS["xgboost"], alpha=0.7)
    axes[0, 0].axhline(0, color="white", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residual XGBoost (Aktual − Prediksi)")
    axes[0, 0].set_ylabel("Residual (Rp)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(dates_lstm, resid_lstm, color=COLORS["lstm"],    alpha=0.7)
    axes[0, 1].axhline(0, color="white", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residual LSTM (Aktual − Prediksi)")
    axes[0, 1].set_ylabel("Residual (Rp)")
    axes[0, 1].grid(True)

    # Histogram residual
    axes[1, 0].hist(resid_xgb,  bins=40, color=COLORS["xgboost"],
                    edgecolor="black", alpha=0.8)
    axes[1, 0].axvline(0, color="white", linestyle="--")
    axes[1, 0].set_title("Distribusi Residual XGBoost")
    axes[1, 0].set_xlabel("Residual (Rp)")
    axes[1, 0].set_ylabel("Frekuensi")
    axes[1, 0].grid(True)

    axes[1, 1].hist(resid_lstm, bins=40, color=COLORS["lstm"],
                    edgecolor="black", alpha=0.8)
    axes[1, 1].axvline(0, color="white", linestyle="--")
    axes[1, 1].set_title("Distribusi Residual LSTM")
    axes[1, 1].set_xlabel("Residual (Rp)")
    axes[1, 1].set_ylabel("Frekuensi")
    axes[1, 1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "11_residual_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Simpan: {out_path}")

    print(f"\n  ✅ Semua visualisasi tersimpan di folder: {OUTPUT_DIR}")


# =============================================================================
# RINGKASAN AKHIR
# =============================================================================

def cetak_ringkasan(df_metrik: pd.DataFrame) -> None:
    """Mencetak ringkasan hasil analisis prediksi."""
    print("\n" + "="*70)
    print("  RINGKASAN HASIL ANALISIS PREDIKSI SAHAM ANTM")
    print("="*70)
    print(df_metrik.to_string())
    print()

    # Tentukan model terbaik berdasarkan RMSE
    model_terbaik = df_metrik["RMSE"].idxmin()
    print(f"  🏆 Model Terbaik (RMSE terendah): {model_terbaik}")
    print(f"     RMSE  = {df_metrik.loc[model_terbaik, 'RMSE']:,.4f}")
    print(f"     MAE   = {df_metrik.loc[model_terbaik, 'MAE']:,.4f}")
    print(f"     MAPE  = {df_metrik.loc[model_terbaik, 'MAPE']:.4f} %")
    print(f"     R²    = {df_metrik.loc[model_terbaik, 'R2']:.4f}")
    print("\n" + "="*70)
    print(f"  Output disimpan di: {OUTPUT_DIR}")
    print("="*70 + "\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "█"*70)
    print("  ANALISIS PREDIKSI HARGA SAHAM ANTM — BEI")
    print("  Model: XGBoost & LSTM")
    print("█"*70)

    # Step 1: Load data
    df = load_data(DATASET_PATH)

    # Step 2: EDA
    exploratory_data_analysis(df)

    # Step 3: Preprocessing
    df_clean = preprocessing_data(df)

    # Pembagian & normalisasi
    data = split_and_scale(df_clean, test_size=TEST_SIZE)

    # Step 4a: Model XGBoost
    xgb_model, y_pred_xgb = bangun_model_xgboost(data)

    # Step 4b: Model LSTM
    lstm_model, lstm_history, y_pred_lstm, n_lstm = bangun_model_lstm(data, window=WINDOW_SIZE)

    # Step 5: Evaluasi
    df_metrik, y_test_xgb, y_test_lstm = evaluasi_model(
        data["y_test"], y_pred_xgb, y_pred_lstm, n_lstm
    )

    # Step 6: Visualisasi
    visualisasi_hasil(data, y_pred_xgb, y_pred_lstm, n_lstm,
                      df_metrik, y_test_xgb, y_test_lstm)

    # Ringkasan
    cetak_ringkasan(df_metrik)

    return xgb_model, lstm_model, df_metrik


if __name__ == "__main__":
    main()