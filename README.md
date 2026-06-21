# Prediksi Harga Saham ANTM Menggunakan XGBoost dan LSTM

Proyek ini bertujuan untuk membandingkan kinerja algoritma Extreme Gradient Boosting (XGBoost) dan Long Short-Term Memory (LSTM) dalam memprediksi harga penutupan saham PT Aneka Tambang Tbk (ANTM) satu hari ke depan.

Penelitian menggunakan data historis saham ANTM yang diperoleh dari Bursa Efek Indonesia melalui dataset Kaggle periode Juli 2019 hingga April 2024 sebanyak 1.120 hari perdagangan.

## Fitur yang Digunakan

### Variabel Utama
- Open Price
- High
- Low
- Close
- Volume
- Value
- Frequency
- Foreign Buy
- Foreign Sell
- Change

### Feature Engineering
- Moving Average (MA 5, 10, 20, 50)
- Exponential Moving Average (EMA)
- MACD
- RSI
- Bollinger Bands
- Volatility
- Lag Features

## Metodologi
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Feature Engineering
4. Normalisasi Data
5. Time-Series Train-Test Split (80:20)
6. Pemodelan XGBoost
7. Pemodelan LSTM
8. Evaluasi Model

## Metrik Evaluasi
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-Squared (R²)

## Hasil

| Model | MAE | RMSE | MAPE | R² |
|---------|---------|---------|---------|---------|
| XGBoost | 52,35 | 69,42 | 3,06% | 0,8624 |
| LSTM | 60,63 | 75,61 | 3,30% | 0,8367 |

## Kesimpulan

Hasil penelitian menunjukkan bahwa model XGBoost memberikan performa terbaik dengan tingkat kesalahan prediksi (MAPE) sebesar 3,06%, sedikit lebih baik dibandingkan LSTM sebesar 3,30%. Variabel harga harian (Open, High, Low, Close) menjadi faktor paling dominan dalam memprediksi harga saham ANTM pada hari berikutnya.