# Deskripsi tugas

Perusahaan taksi bernama Sweet Lift telah mengumpulkan data historis tentang pesanan taksi di bandara. Untuk menarik lebih banyak pengemudi pada jam sibuk, perlu memprediksi jumlah pesanan taksi untuk satu jam berikutnya. Buat model untuk prediksi seperti itu.

Metrik RMSE pada *test set* tidak boleh lebih dari 48.

## Instruksi tugas

1. Unduh data dan lakukan *resampling* dalam satu jam.
2. Analisis datanya.
3.  Latih model yang berbeda dengan hiperparameter yang berbeda pula. Sampel tes harus 10% dari *dataset* awal.
4. Uji data menggunakan sampel tes dan berikan kesimpulan.

## Deskripsi data

Data tersimpan di file `taxi.csv`. Jumlah pesanan di kolom'*num_orders*'.

## Persiapan


```python
# import pandas and numpy untuk proses dan manipulasi data
import numpy as np
import pandas as pd

# matplotlib dan seaborn untuk statistika data visualisasi
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# import statistik model
from statsmodels.tsa.seasonal import seasonal_decompose

# import acf auto correlation dan pacf parsial auto correlation untuk grafik time series 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# i impor modul untuk pemisahan dan validasi silang menggunakan pencarian grid
from sklearn.model_selection import train_test_split, GridSearchCV

# import time series 
from sklearn.model_selection import TimeSeriesSplit

# impor metrik untuk mengukur kualitas model
from sklearn.metrics import mean_squared_error

# impor model pembelajaran mesin
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool 
from lightgbm import LGBMRegressor 
from xgboost import XGBRegressor 
from sklearn.neighbors import KNeighborsRegressor 

from IPython.display import display

# abaikan peringatan
import warnings
warnings.filterwarnings("ignore")

print('Perpustakaan proyek telah berhasil diimpor!')
```

    Perpustakaan proyek telah berhasil diimpor!


# Memuat Data dari csv agar dapat dijalankan dengan pandas untuk menjadi DataFrame


```python
data = pd.read_csv('/datasets/taxi.csv')
```


```python
# Membuat Fungsi untuk menentukan jika ada columns yang memiliki nilai yang hilang
def get_percent_of_na(df, num):
    count = 0
    df = df.copy()
    s = (df.isna().sum() / df.shape[0])
    for column, percent in zip(s.index, s.values):
        num_of_nulls = df[column].isna().sum()
        if num_of_nulls == 0:
            continue
        else:
            count += 1
        print('Column {} dengan {:.{}%} persentasi nilai yang hilang , dan {} nilai yang hilang'.format(column, percent, num, num_of_nulls))
    if count != 0:
        print("\033[1m" + 'Terdapat {} columns dengan nilai NA.'.format(count) + "\033[0m")
    else:
        print()
        print("\033[1m" + 'Tidak Terdapat columns dengan nilai NA.' + "\033[0m")
        
# Fungsi untuk melihat informasi keseluruhan pada dataset 
def get_info(df):
    print("\033[1m" + '-'*100 + "\033[0m")
    print('Head:')
    print()
    display(df.head(25))
    print('-'*100)
    print('Info:')
    print()
    display(df.info())
    print('-'*100)
    print('Describe:')
    print()
    display(df.describe())
    print('-'*100)
    display(df.describe(include='object'))
    print()
    print('Columns dengan nilai yang hilang:')
    display(get_percent_of_na(df, 4))  
    print('-'*100)
    print('Shape:')
    print(df.shape)
    print('-'*100)
    print('Duplicated:')
    print("\033[1m" + 'Kita mempunyai {} baris yang terduplikasi.\n'.format(df.duplicated().sum()) + "\033[0m")
    print()
```


```python
print('Informasi Umum pada Dataset')
get_info(data)
```

    Informasi Umum pada Dataset
    [1m----------------------------------------------------------------------------------------------------[0m
    Head:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>num_orders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-03-01 00:00:00</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-03-01 00:10:00</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-03-01 00:20:00</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-03-01 00:30:00</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-03-01 00:40:00</td>
      <td>32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-03-01 00:50:00</td>
      <td>21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-03-01 01:00:00</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-03-01 01:10:00</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-03-01 01:20:00</td>
      <td>17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-03-01 01:30:00</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018-03-01 01:40:00</td>
      <td>19</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018-03-01 01:50:00</td>
      <td>25</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2018-03-01 02:00:00</td>
      <td>22</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018-03-01 02:10:00</td>
      <td>12</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-03-01 02:20:00</td>
      <td>19</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2018-03-01 02:30:00</td>
      <td>8</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2018-03-01 02:40:00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-03-01 02:50:00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018-03-01 03:00:00</td>
      <td>8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2018-03-01 03:10:00</td>
      <td>17</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018-03-01 03:20:00</td>
      <td>7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018-03-01 03:30:00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2018-03-01 03:40:00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2018-03-01 03:50:00</td>
      <td>20</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2018-03-01 04:00:00</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    Info:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26496 entries, 0 to 26495
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   datetime    26496 non-null  object
     1   num_orders  26496 non-null  int64 
    dtypes: int64(1), object(1)
    memory usage: 414.1+ KB



    None


    ----------------------------------------------------------------------------------------------------
    Describe:
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_orders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26496.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.070463</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.211330</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>119.000000</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26496</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>26496</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2018-07-12 06:50:00</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Columns dengan nilai yang hilang:
    
    [1mTidak Terdapat columns dengan nilai NA.[0m



    None


    ----------------------------------------------------------------------------------------------------
    Shape:
    (26496, 2)
    ----------------------------------------------------------------------------------------------------
    Duplicated:
    [1mKita mempunyai 0 baris yang terduplikasi.
    [0m
    


Dari hasil informasi keseluruhan umum data awal terdapat 26496 baris dan 2 kolom data, tidak terdapat nilai yang hilang pada data serta tidak terdapat duplikasi pada data, namun kita perlu merubah kolom `datetime` dari object ke tipe data datetime dan menurunkan kolom `num_orders` dari int64 ke int32 untuk mengurangu penggunaan memori. 

# Mengubah tipe data


```python
# mengubah tipe data menjadi tipedata yang tepat
def change_datatype(df, cols, type_val):
    for col in cols:
        df[col] = df[col].astype(type_val)
        
change_datatype(data, ['datetime'], 'datetime64[ns]')
change_datatype(data, ['num_orders'], 'int32')
    
```


```python
# change data to the right type
def change_datatype(df, cols, type_val):
    for col in cols:
        df[col] = df[col].astype(type_val)
        
change_datatype(data, ['datetime'], 'datetime64[ns]')
change_datatype(data, ['num_orders'], 'int32')
```


```python
# memeriks info dataset
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26496 entries, 0 to 26495
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    26496 non-null  datetime64[ns]
     1   num_orders  26496 non-null  int32         
    dtypes: datetime64[ns](1), int32(1)
    memory usage: 310.6 KB


Kita telah berhasil mengubah tipe data dari kolom `datetime` menjadi datetime64 dan `num_orders` menjadi int32. pada tahap selanjutnya kita akan memeriksa urutan waktu dan tanggal telah sesuai dengan urutan kronologisnya dengan menggunakan atribut `is_monotonic`.


```python
# mengubah tabel index ke datetime
taxi_df = data.set_index('datetime')
taxi_df.sort_index(axis=0, inplace=True)

# menampilkan index data
taxi_df.index
```




    DatetimeIndex(['2018-03-01 00:00:00', '2018-03-01 00:10:00',
                   '2018-03-01 00:20:00', '2018-03-01 00:30:00',
                   '2018-03-01 00:40:00', '2018-03-01 00:50:00',
                   '2018-03-01 01:00:00', '2018-03-01 01:10:00',
                   '2018-03-01 01:20:00', '2018-03-01 01:30:00',
                   ...
                   '2018-08-31 22:20:00', '2018-08-31 22:30:00',
                   '2018-08-31 22:40:00', '2018-08-31 22:50:00',
                   '2018-08-31 23:00:00', '2018-08-31 23:10:00',
                   '2018-08-31 23:20:00', '2018-08-31 23:30:00',
                   '2018-08-31 23:40:00', '2018-08-31 23:50:00'],
                  dtype='datetime64[ns]', name='datetime', length=26496, freq=None)




```python
# menampilkan baris awal dari data yang di sort
taxi_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_orders</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-03-01 00:00:00</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2018-03-01 00:10:00</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2018-03-01 00:20:00</th>
      <td>28</td>
    </tr>
    <tr>
      <th>2018-03-01 00:30:00</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2018-03-01 00:40:00</th>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# memeriksa apakaj tanggal dan waktu sudah sesuai dengan kronologi pesanannya
print(taxi_df.index.is_monotonic)
print()
print(taxi_df.info())
```

    True
    
    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 26496 entries, 2018-03-01 00:00:00 to 2018-08-31 23:50:00
    Data columns (total 1 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   num_orders  26496 non-null  int32
    dtypes: int32(1)
    memory usage: 310.5 KB
    None



```python
# nilai minimum pada data 
print('Minimum timestamp', taxi_df.index.min())
print()
# nilai maximum pada data
print('Maximum timestamp', taxi_df.index.max())
```

    Minimum timestamp 2018-03-01 00:00:00
    
    Maximum timestamp 2018-08-31 23:50:00


Selanjutnya kita akan memvisualisasikan data :



```python
# menampilkan visualisasi dari time serie

ts = taxi_df['num_orders']
plt.figure(figsize=(15,8))
plt.title('Nomor Pesanan Taksi dari Sweet Lift Taxi Company')
plt.xlabel('Waktu')
plt.ylabel('No. Pesanan')
plt.plot(ts);
```


    
![png](output_18_0.png)
    


Plot di atas adalah plot time series jumlah pesanan taksi Sweet Lift Taxi Company antara 1 Maret 2018 hingga 31 Agustus 2018. Melihat plot tersebut, kita bisa melihat tren pada data kita. Artinya, kita dapat menggunakan deret waktu untuk memodelkan data dan menghasilkan prakiraan. Kita dapat menganalisis data menggunakan berbagai komponen deret waktu.

# Resampling dalam 1 jam 

disini kita akan mengubah interval data timeseries pada kasus kali ini dengan resampling 1 jam 


```python
# resampling data 1 jam 
ts = ts.resample('1H').sum()
ts
```




    datetime
    2018-03-01 00:00:00    124
    2018-03-01 01:00:00     85
    2018-03-01 02:00:00     71
    2018-03-01 03:00:00     66
    2018-03-01 04:00:00     43
                          ... 
    2018-08-31 19:00:00    136
    2018-08-31 20:00:00    154
    2018-08-31 21:00:00    159
    2018-08-31 22:00:00    223
    2018-08-31 23:00:00    205
    Freq: H, Name: num_orders, Length: 4416, dtype: int32




```python
# visualisasi  plot interval data pada 1 jam 
resampling1H = seasonal_decompose(ts)
resampling1H.plot()
plt.show()
```


    
![png](output_23_0.png)
    


`Kesimpulan`  

Kita telah menyiapkan data dengan mengubah tipe data ke tipe yang tepat,  kita juga telah mengurutkan data dan memeriksa apakah diurutkan secara kronologis serta memplot visualisasi dari data yang menunjukkan jumlah pesanan berdasarkan waktu. Kami kemudian mengubah deret waktu menjadi 1 jam. Sekarang kita dapat mulai menganalisis data.

## Analisis

Pada tahap ini kita dapat mengidentifikasi tren dalam data dengan menghaluskan deret waktu dari data awal. kita akan menerapkan `rolling mean` atau `moving average` untuk mengurangi fluktuasi dalam deret waktu.

# Rolling Mean 


```python
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    
    """
    seris = dataframe dengan timeseries
    window = pergerakan ukuran window
    plot_intervals = menampilkan confident interval data
    plot_anomalies = menampilkan anomali pada data
    
    """
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(15,5))
    plt.title("Rata-rata bergerak(Moving Average)\n ukuran window = {}".format(window))
    plt.plot(rolling_mean, "g", label="Tren dari Pergerakan Rata-rata (Rolling Mean)")
    
    # Plot confidence intervals untuk mmenghaluskan deret waktu
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        
        
    # memiliki interval, temukan nilai abnormal
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plr.plot(anomalies, "ro", markersize=10)
            
        
    plt.plot(series[window:], label="Nilai yang sebenarnya")
    plt.legend(loc="upper left")
    plt.grid(True)
```

Jika kita menghaluskan data dengan menggunakan ukuran window 30 untuk resample menjadi `1 Hari`, kita akan memiliki plot di bawah ini. Dari plot tersebut, kita bisa melihat peningkatan jumlah taksi yang dipesan setiap hari.


```python
# plot rolling mean 
ts_ = ts.resample('1D').sum()
plotMovingAverage(ts_, 30)
```


    
![png](output_30_0.png)
    


Dari plot rolling mean di atas untuk ukuran jendela 30, kita dapat mengamati peningkatan yang stabil dalam jumlah pesanan dari April hingga September 2018. Plot deret waktu diatas juga  menunjukkan peningkatan tren, dimana menunjukkan peningkatan pesanan jangka panjang selama periode April hingga September 2018.

# Tren dan Musiman


```python
decomposed = seasonal_decompose(ts_)

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca(), figsize=(15, 10))
plt.title('Tren');
```


    
![png](output_33_0.png)
    


Kami mengamati bahwa data menunjukkan tren naik. Dengan menggunakan garis tren, kita dapat membuat peramalan ke masa depan.


```python
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca(), figsize=(15, 10))
plt.title('Seasonality');
```


    
![png](output_35_0.png)
    


Plot di atas menunjukkan fluktuasi periodik dalam deret waktu dalam periode tertentu. Fluktuasi tersebut membentuk pola yang cenderung berulang dari satu periode musim ke musim berikutnya.


```python
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca(), figsize=(15, 10))
plt.title('Residuals')
plt.tight_layout()
```


    
![png](output_37_0.png)
    


Mari kita lihat perbedaan pesanan selama akhir pekan tertentu. Kita akan memplot grafik komponen musiman selama dua minggu pertama di bulan Maret 2018.


```python
# plot seasonal pada 15 hari pertama di bulan maret
decomposed.seasonal['2018-03-01' : '2018-03-15'].plot(figsize=(8,6))
plt.title('plot seasonal pada 15 hari pertama di bulan maret')
plt.show()
```


    
![png](output_39_0.png)
    


Dari plot di atas, kita dapat mengamati Pemesanan taksi mencapai puncaknya pada tanggal 2 (Senin), 5 (Jumat), 9 (Senin), dan 12 (Jumat) Maret 2018. Pola yang dibentuk oleh fluktuasi ini berulang dari satu periode musim ke periode berikutnya.

# Autocorrelation dan Partial autocorrelation

Kita dapat menerapkan autokorelasi yanng mana merupakan korelasi antara dua nilai dalam deret waktu untuk menemukan pola dalam data kita. Kita akan menggunakan fungsi autokorelasi (ACF) untuk mengidentifikasi kelambatan (lag) mana yang memiliki korelasi signifikan, memahami pola dan properti deret waktu, lalu menggunakan informasi tersebut untuk memodelkan data deret waktu. Dari ACF, kita dapat menilai keacakan dan stasioneritas deret waktu, dan juga menentukan apakah ada tren dan pola musiman.


```python
# plot fungsi autocorrelation 
fig, ax = plt.subplots(figsize=(12,6))
plot_acf(ts_, lags= 40, title= 'Fungsi Autocorrelation untuk pesanan taxi\n (dengan 5% tingkat signifikansi dari autokorelasi)', ax=ax)
plt.ylabel('Autokorelasi')
plt.xlabel('Lag')
plt.show()
```


    
![png](output_43_0.png)
    


Dalam plot ACF, setiap batang mewakili ukuran dan arah korelasi. Kita dapat mengamati bahwa tren hadir dalam deret waktu, kelambatan yang lebih pendek biasanya memiliki korelasi positif yang besar karena pengamatan yang lebih dekat dalam waktu cenderung memiliki nilai yang sama. Korelasi meruncing perlahan saat kelambatan meningkat. Batang yang melintasi garis biru signifikan secara statistik. Dalam plot ACF untuk pesanan taksi, autokorelasi menurun secara perlahan. Lima belas kelambatan pertama sangat signifikan.


```python
# plot dar partial autocorrelation function
fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(ts_, lags = 40, title='Fungsi autokorelasi parsial untuk pesanan taksi', ax=ax)
# Anotasi PACF di k
ax.annotate('PACF di k = 0', 
            xy=(0.5, 1.0), 
            xycoords='data', 
            fontsize=10, 
            xytext=(50, 0),
            textcoords='offset points', 
            arrowprops=dict(arrowstyle='->', color='red'), 
            horizontalalignment='center',  
            verticalalignment='center')
ax.annotate(
    'PACF di k = 1', 
    xy=(1.5, 0.9), 
    xycoords='data', 
    fontsize=10, 
    xytext=(50, 0), 
    textcoords='offset points', 
    arrowprops=dict(arrowstyle='->', color='red'), 
    horizontalalignment='center',  
    verticalalignment='center')
ax.annotate(
    'PACF di k = 2', 
    xy=(2.5, 0.35), 
    xycoords='data', 
    fontsize=10, 
    xytext=(50, 0), 
    textcoords='offset points', 
    arrowprops=dict(arrowstyle='->', color='red'), 
    horizontalalignment='center',  
    verticalalignment='center')
plt.ylabel('Partial Autocorrelation')
plt.xlabel('Lag')
plt.show()
```


    
![png](output_45_0.png)
    


PACF pada Lag 0 adalah 1,0. PACF pada Lag 1 adalah 0,9. Nilai PACF pada Lag 2 adalah 0,3. Pada grafik, autokorelasi parsial untuk kelambatan 0, 1 dan 2 secara statistik signifikan. Kelambatan berikutnya hampir signifikan. Dengan menilai pola autokorelasi dan autokorelasi parsial dalam data, kita dapat memahami sifat deret waktu kita dan memodelkannya.

## Pelatihan

Pada tahap ini kita akan membuat fitur baru untuk data deret waktu. Kita dapat membuat fitur bulan, hari, hari dalam seminggu dan jam dari kolom tanggal dan waktu. Kami juga akan membuat kelambatan (lag)


```python
# fungsi untuk membuat features baru
def make_features(data, max_lag, rolling_mean_size):
    data['month']     = data.index.month
    data['day']       = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['hour']      = data.index.hour
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)
        
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()
```


```python
# membuat features baru
ts = pd.DataFrame(ts)
make_features(ts,6,7)
ts.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_orders</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>hour</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
      <th>lag_6</th>
      <th>rolling_mean</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-03-01 00:00:00</th>
      <td>124</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 01:00:00</th>
      <td>85</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>124.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 02:00:00</th>
      <td>71</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>85.0</td>
      <td>124.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 03:00:00</th>
      <td>66</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>71.0</td>
      <td>85.0</td>
      <td>124.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 04:00:00</th>
      <td>43</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>85.0</td>
      <td>124.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 05:00:00</th>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>85.0</td>
      <td>124.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 06:00:00</th>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>85.0</td>
      <td>124.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-03-01 07:00:00</th>
      <td>15</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>85.0</td>
      <td>58.142857</td>
    </tr>
    <tr>
      <th>2018-03-01 08:00:00</th>
      <td>34</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>42.571429</td>
    </tr>
    <tr>
      <th>2018-03-01 09:00:00</th>
      <td>69</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>35.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 10:00:00</th>
      <td>64</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>10</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>2018-03-01 11:00:00</th>
      <td>96</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>11</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>34.714286</td>
    </tr>
    <tr>
      <th>2018-03-01 12:00:00</th>
      <td>30</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>42.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 13:00:00</th>
      <td>32</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>13</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>45.714286</td>
    </tr>
    <tr>
      <th>2018-03-01 14:00:00</th>
      <td>48</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>14</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>48.571429</td>
    </tr>
    <tr>
      <th>2018-03-01 15:00:00</th>
      <td>66</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>15</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>53.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 16:00:00</th>
      <td>43</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>57.857143</td>
    </tr>
    <tr>
      <th>2018-03-01 17:00:00</th>
      <td>44</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>17</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>54.142857</td>
    </tr>
    <tr>
      <th>2018-03-01 18:00:00</th>
      <td>73</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>18</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>51.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 19:00:00</th>
      <td>45</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>19</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>2018-03-01 20:00:00</th>
      <td>61</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>20</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>50.142857</td>
    </tr>
    <tr>
      <th>2018-03-01 21:00:00</th>
      <td>66</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>21</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>54.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 22:00:00</th>
      <td>113</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>22</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>56.857143</td>
    </tr>
    <tr>
      <th>2018-03-01 23:00:00</th>
      <td>58</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>63.571429</td>
    </tr>
    <tr>
      <th>2018-03-02 00:00:00</th>
      <td>90</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>65.714286</td>
    </tr>
    <tr>
      <th>2018-03-02 01:00:00</th>
      <td>120</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>72.285714</td>
    </tr>
    <tr>
      <th>2018-03-02 02:00:00</th>
      <td>75</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>2018-03-02 03:00:00</th>
      <td>64</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>83.285714</td>
    </tr>
    <tr>
      <th>2018-03-02 04:00:00</th>
      <td>20</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>83.714286</td>
    </tr>
    <tr>
      <th>2018-03-02 05:00:00</th>
      <td>11</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>20.0</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>77.142857</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop nilai NaN dari time series data
ts = ts.dropna()
print('Dataset Time Series Mempunyai', ts.shape[0], 'baris dan', ts.shape[1], 'features')
print()
ts.head(30)
```

    Dataset Time Series Mempunyai 4409 baris dan 12 features
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_orders</th>
      <th>month</th>
      <th>day</th>
      <th>dayofweek</th>
      <th>hour</th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
      <th>lag_6</th>
      <th>rolling_mean</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-03-01 07:00:00</th>
      <td>15</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>85.0</td>
      <td>58.142857</td>
    </tr>
    <tr>
      <th>2018-03-01 08:00:00</th>
      <td>34</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>71.0</td>
      <td>42.571429</td>
    </tr>
    <tr>
      <th>2018-03-01 09:00:00</th>
      <td>69</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>35.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 10:00:00</th>
      <td>64</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>10</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>43.0</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>2018-03-01 11:00:00</th>
      <td>96</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>11</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>34.714286</td>
    </tr>
    <tr>
      <th>2018-03-01 12:00:00</th>
      <td>30</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>42.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 13:00:00</th>
      <td>32</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>13</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>15.0</td>
      <td>45.714286</td>
    </tr>
    <tr>
      <th>2018-03-01 14:00:00</th>
      <td>48</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>14</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>34.0</td>
      <td>48.571429</td>
    </tr>
    <tr>
      <th>2018-03-01 15:00:00</th>
      <td>66</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>15</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>69.0</td>
      <td>53.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 16:00:00</th>
      <td>43</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>57.857143</td>
    </tr>
    <tr>
      <th>2018-03-01 17:00:00</th>
      <td>44</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>17</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>96.0</td>
      <td>54.142857</td>
    </tr>
    <tr>
      <th>2018-03-01 18:00:00</th>
      <td>73</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>18</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>30.0</td>
      <td>51.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 19:00:00</th>
      <td>45</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>19</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>2018-03-01 20:00:00</th>
      <td>61</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>20</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>50.142857</td>
    </tr>
    <tr>
      <th>2018-03-01 21:00:00</th>
      <td>66</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>21</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>66.0</td>
      <td>54.285714</td>
    </tr>
    <tr>
      <th>2018-03-01 22:00:00</th>
      <td>113</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>22</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>43.0</td>
      <td>56.857143</td>
    </tr>
    <tr>
      <th>2018-03-01 23:00:00</th>
      <td>58</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>44.0</td>
      <td>63.571429</td>
    </tr>
    <tr>
      <th>2018-03-02 00:00:00</th>
      <td>90</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>73.0</td>
      <td>65.714286</td>
    </tr>
    <tr>
      <th>2018-03-02 01:00:00</th>
      <td>120</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>45.0</td>
      <td>72.285714</td>
    </tr>
    <tr>
      <th>2018-03-02 02:00:00</th>
      <td>75</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>61.0</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>2018-03-02 03:00:00</th>
      <td>64</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>66.0</td>
      <td>83.285714</td>
    </tr>
    <tr>
      <th>2018-03-02 04:00:00</th>
      <td>20</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>113.0</td>
      <td>83.714286</td>
    </tr>
    <tr>
      <th>2018-03-02 05:00:00</th>
      <td>11</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>20.0</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>58.0</td>
      <td>77.142857</td>
    </tr>
    <tr>
      <th>2018-03-02 06:00:00</th>
      <td>11</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>6</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>90.0</td>
      <td>62.571429</td>
    </tr>
    <tr>
      <th>2018-03-02 07:00:00</th>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>120.0</td>
      <td>55.857143</td>
    </tr>
    <tr>
      <th>2018-03-02 08:00:00</th>
      <td>46</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>64.0</td>
      <td>75.0</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>2018-03-02 09:00:00</th>
      <td>45</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>46.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>64.0</td>
      <td>33.428571</td>
    </tr>
    <tr>
      <th>2018-03-02 10:00:00</th>
      <td>54</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>10</td>
      <td>45.0</td>
      <td>46.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>29.142857</td>
    </tr>
    <tr>
      <th>2018-03-02 11:00:00</th>
      <td>91</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>54.0</td>
      <td>45.0</td>
      <td>46.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>27.714286</td>
    </tr>
    <tr>
      <th>2018-03-02 12:00:00</th>
      <td>36</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>12</td>
      <td>91.0</td>
      <td>54.0</td>
      <td>45.0</td>
      <td>46.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>37.857143</td>
    </tr>
  </tbody>
</table>
</div>



# Memisahkan data train dan test dari 10% data awal 


```python
# Fungsi untuk mendefenisikan metric
def rmse(true, pred):
    return mean_square_error(true, pred)**0.5
```


```python
# memisahkan data train dan test set
train, test = train_test_split(ts, shuffle=False, test_size=0.1)
```


```python
# menampilkan index dari data set
print(train.index.min(),'sampai dengan', train.index.max())
print(test.index.min(),'sampai dengan',  test.index.max())
print()
print('Train set mempunyai', train.shape[0], 'dan baris', train.shape[1], 'features')
print('Test set mempunyai', test.shape[0], 'dan baris', test.shape[1], 'features')

```

    2018-03-01 07:00:00 sampai dengan 2018-08-13 14:00:00
    2018-08-13 15:00:00 sampai dengan 2018-08-31 23:00:00
    
    Train set mempunyai 3968 dan baris 12 features
    Test set mempunyai 441 dan baris 12 features



```python
# menentukan variable untuk target dan features
features_train = train.drop(['num_orders'], axis=1)
target_train = train['num_orders']

features_test = test.drop(['num_orders'], axis=1)
target_test = test['num_orders']
```

Dalam melatih dan menguji model, kita akan mencoba algoritme berikut :


1. Linear Regression
2. Random Forest Regression
3. Catboost Regressor
4. XGBoost Regressor
5. LightGBM Regressor
6. KNeighbors Regressor


Untuk menyetel hyperparameter, kami menggunakan pemisahan deret waktu untuk validasi silang


```python
# memisahkan time series 
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)
```

    TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)


# Linear Regression


```python
# Fungsi untuk melatih model dan membuat prediksi 
def train_linear_model(X_train, y_train):
    """fungsi ini untuk melatih regresi linear"""
    global lr_model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
def linear_regressor_pred(X_test, y_test):
    """
    Fungsi untuk membuat prediksi
    menggunakan regresi linear
    """
    
    lr_pred = lr_model.predict(X_test)
    # RMSE
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print("\033[1m" + 'RMSE dari Regresi Linear' + "\033[0m")
    print('RMSE: {:.3f}'.format(lr_rmse))
    print()
```


```python
%%time
# melatih model
train_linear_model(features_train, target_train)
```

    CPU times: user 5.2 ms, sys: 274 µs, total: 5.48 ms
    Wall time: 3.99 ms



```python
%%time
# membuat prediksi dan RMSE
linear_regressor_pred(features_test, target_test)
```

    [1mRMSE dari Regresi Linear[0m
    RMSE: 52.740
    
    CPU times: user 2.71 ms, sys: 7.77 ms, total: 10.5 ms
    Wall time: 3.41 ms


# KNeighbors Regressor


```python
%%time
# optimasi hyperparameter model KNeighbors regresi

# menentukan tuning hyper parameter
knn_grid = {'n_neighbors' : range(1,5,1),
           'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
           }
# model 
knn_reg = KNeighborsRegressor()

# Grid search 
grid_search_knn = GridSearchCV(
    estimator   = knn_reg,
    param_grid  = knn_grid,
    scoring     = "neg_mean_squared_error",
    cv          = tscv, 
    n_jobs      = 1
)

# menjalankan grid search
grid_search_knn.fit(features_train, target_train)

# hasil 
print('Hyperparameter terbaik: {}'.format(grid_search_knn.best_params_))
```

    Hyperparameter terbaik: {'algorithm': 'auto', 'n_neighbors': 4}
    CPU times: user 2.13 s, sys: 1.51 s, total: 3.65 s
    Wall time: 3.69 s



```python
# fungsi melatih model dan prediksi 
def train_KNeighbors_regressor(X_train, y_train):
    """fungsi ini untuk melatih KNeighbors regressor model"""
    global knn_model
    # model
    knn_model = KNeighborsRegressor(**grid_search_knn.best_params_)
    knn_model.fit(X_train, y_train)
        
def KNeighbors_regressor_pred(X_test, y_test):
    """
    Fungsi untuk membuat prediksi
    menggunakan KNeighbors regression
    """
    # model
    knn_pred = knn_model.predict(X_test)
    # rmse
    knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
    print("\033[1m" + 'RMSE dari KNeighbors regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(knn_rmse))
    print()
```


```python
%%time
train_KNeighbors_regressor(features_train, target_train)
```

    CPU times: user 9.57 ms, sys: 580 µs, total: 10.1 ms
    Wall time: 9.13 ms



```python
%%time 
KNeighbors_regressor_pred(features_test, target_test)
```

    [1mRMSE dari KNeighbors regressor[0m
    RMSE: 62.147
    
    CPU times: user 15.7 ms, sys: 0 ns, total: 15.7 ms
    Wall time: 13.9 ms


# Random Forest Regressor


```python
%%time
# optimasi hyperparameter model Random Forest Regressor

# menentukan tuning hyper parameter
grid = {
    "n_estimators"       : [10, 25, 50, 100],
    "max_depth"          : [None, 2, 4, 8, 10, 12],
    "min_samples_leaf"   : [2, 4, 6]
}

# model 
regressor = RandomForestRegressor(random_state = 12345)
# grid search
grid_search_rf = GridSearchCV(estimator = regressor, param_grid = grid, scoring= 'neg_mean_squared_error', cv=tscv)
# menjalankan grid search
grid_search_rf.fit(features_train, target_train)
# hasil 
print('Parameter terbaik : {}'. format(grid_search_rf.best_params_))
```

    Parameter terbaik : {'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 100}
    CPU times: user 2min 11s, sys: 320 ms, total: 2min 12s
    Wall time: 2min 12s



```python
# fungsi melatih model dan prediksi
def train_random_forest(X_train, y_train):
    """fungsi ini untuk melatih random forest regressor """
    global rf_model
    # model
    rf_model = RandomForestRegressor(**grid_search_rf.best_params_)
    rf_model.fit(X_train, y_train)
    
def random_forest_pred(X_test, y_test):
    """
    Fungsi untuk membuat prediksi
    menggunakan random forest regressor
    """
    # model 
    rf_pred = rf_model.predict(X_test)
    # rmse
    rf_pred = np.sqrt(mean_squared_error(y_test, rf_pred))
    print("\033[1m" + 'RMSE dari Random Forest Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(rf_pred))
    print()
    
    # features penting dari model 
    sorted_feature_importance = rf_model.feature_importances_. argsort()
    plt.figure(figsize=(8,6))
    plt.barh(features_train.columns[sorted_feature_importance],
            rf_model.feature_importances_[sorted_feature_importance],
             color='turquoise'
            )
    plt.xlabel("Feature penting dari Random Forest ")
```


```python
%%time
train_random_forest(features_train, target_train)
```

    CPU times: user 2.38 s, sys: 3.94 ms, total: 2.39 s
    Wall time: 2.4 s



```python
%%time
random_forest_pred(features_test, target_test)
```

    [1mRMSE dari Random Forest Regressor[0m
    RMSE: 44.786
    
    CPU times: user 73.9 ms, sys: 41 µs, total: 73.9 ms
    Wall time: 71.3 ms



    
![png](output_72_1.png)
    


# Catboost Regressor


```python
%%time
# optimasi hyperparameter model catboost regressor

# menentukan tuning hyper parameter
grid= {'learning_rate'  : [0.001, 0.01, 0.5],
       'depth'          : [4, 6, 10],
       'l2_leaf_reg'    : [1, 3, 5, 7, 9]
      }

# model
cb_reg = CatBoostRegressor(
    iterations            = 500,
    logging_level          = 'Silent',
    loss_function          = 'RMSE',
    early_stopping_rounds  = 50,
    random_state           = 12345)
# Grid Search 
grid_search = GridSearchCV(estimator = cb_reg, param_grid = grid, scoring='neg_mean_squared_error', cv=tscv)
# menjalankan grid search
grid_search.fit(features_train, target_train)
# hasil
print('Parameter terbaik: {}'. format(grid_search.best_params_))
```

    Parameter terbaik: {'depth': 10, 'l2_leaf_reg': 1, 'learning_rate': 0.01}
    CPU times: user 12min 5s, sys: 2.58 s, total: 12min 8s
    Wall time: 12min 28s



```python
# fungsi melatih model dan prediksi
def train_catboost_regressor(X_train, y_train, X_test, y_test):
    """fungsi ini untuk melatih model catboost regressor"""
    global cb_model
    # model
    cb_model = CatBoostRegressor(**grid_search.best_params_)
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False, plot=False)
    
def catboost_regressor_pred(X_test, y_test):
    """
    Fungsi untuk membuat prediksi
    menggunakan model catboost regressor
    """
    # model
    cb_pred = cb_model.predict(X_test)
    # rmse
    cb_rmse = np.sqrt(mean_squared_error(y_test, cb_pred))
    print("\033[1m" + 'RMSE dari Catboost Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(cb_rmse))
    print()
    
    # features penting dari model
    sorted_feature_importance = cb_model.feature_importances_.argsort()
    plt.figure(figsize=(8,6))
    plt.barh(features_train.columns[sorted_feature_importance],
            cb_model.feature_importances_[sorted_feature_importance],
            color='turquoise')
    plt.xlabel("Feature penting dari Catboost")
```


```python
%%time
train_catboost_regressor(features_train, target_train,features_test, target_test)
```

    CPU times: user 20.9 s, sys: 76 ms, total: 21 s
    Wall time: 21.1 s



```python
%%time
catboost_regressor_pred(features_test, target_test)
```

    [1mRMSE dari Catboost Regressor[0m
    RMSE: 47.622
    
    CPU times: user 30 ms, sys: 5.47 ms, total: 35.4 ms
    Wall time: 31 ms



    
![png](output_77_1.png)
    


# XGBoost Regressor


```python
%%time
# optimasi hyperparameter model xgboost

# menentukan tuning hyper parameter
xgb_grid = {'learning_rate': [0.001, 0.01, 0.1, 0.3], 
            'max_depth': [2, 4, 6, 10],
            'n_estimators': [50, 100, 200, 500]
           }
# model 
xgb_regr = XGBRegressor(random_state = 12345)

# grid search
grid_search_xgb = GridSearchCV(
    estimator = xgb_regr, 
    param_grid = xgb_grid, 
    scoring = "neg_mean_squared_error", 
    cv = tscv, 
    n_jobs = 1
)
# menjalankan grid search
grid_search_xgb.fit(features_train, target_train)
# hasil 
print('Parameter terbaik : {}'.format(grid_search_xgb.best_params_))
```

    Parameter terbaik : {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}
    CPU times: user 13min 45s, sys: 7.25 s, total: 13min 52s
    Wall time: 13min 59s



```python
# fungsi melatih model dan prediksi
def train_xgboost_reg(X_train, y_train):
    """fungsi ini untuk melatih model XGBoost """
    global xgb_model
    # model
    xgb_model = XGBRegressor(**grid_search_xgb.best_params_)
    xgb_model.fit(X_train, y_train) 
    
def xgboost_regressor_pred(X_test, y_test):
    """
    Fungsi untuk membuat prediksi
    menggunakan model XGBoost
    """
    # model
    xgb_pred = xgb_model.predict(X_test)
    # rmse
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    print("\033[1m" + 'RMSE dari XGBoost Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(xgb_rmse))
    print()
    
    # features penting dari model 
    sorted_feature_importance = xgb_model.feature_importances_.argsort()
    plt.figure(figsize=(8,6))
    plt.barh(features_train.columns[sorted_feature_importance], 
             xgb_model.feature_importances_[sorted_feature_importance], 
             color='turquoise')
    plt.xlabel("Feature penting dari XGBoost")
```


```python
%%time
train_xgboost_reg(features_train, target_train)
```

    CPU times: user 1.19 s, sys: 12.2 ms, total: 1.2 s
    Wall time: 1.21 s



```python
%%time
xgboost_regressor_pred(features_test, target_test)
```

    [1mRMSE dari XGBoost Regressor[0m
    RMSE: 45.933
    
    CPU times: user 39.8 ms, sys: 33 µs, total: 39.8 ms
    Wall time: 33.1 ms



    
![png](output_82_1.png)
    


# LightGBM Regressor


```python
%%time
# optimasi hyperparameter model xgboost

# menentukan tuning hyper parameter
lgbm_grid = {'learning_rate': [0.001, 0.01, 0.05, 0.1],
             'n_estimators': [50, 100, 500],
             'num_leaves': [5, 10, 20, 31]
            }

# model 
lgbm_regr = LGBMRegressor(random_state = 12345)

# grid search
grid_search_lgbm = GridSearchCV(
    estimator = lgbm_regr, 
    param_grid = lgbm_grid, 
    scoring = "neg_mean_squared_error", 
    cv = tscv, 
    n_jobs = 1
)

# menjalankan grid search
grid_search_lgbm.fit(features_train, target_train)
# hasil 
print('Parameter terbaik : {}'.format(grid_search_lgbm.best_params_))
```

    Parameter terbaik : {'learning_rate': 0.01, 'n_estimators': 500, 'num_leaves': 31}
    CPU times: user 1min 40s, sys: 1.99 s, total: 1min 42s
    Wall time: 1min 43s



```python
# fungsi melatih model dan prediksi
def train_lightGBM_reg(X_train, y_train):
    """fungsi ini untuk melatih model LightGBM regressor"""
    global lgbm_model
    # model
    lgbm_model = LGBMRegressor(**grid_search_lgbm.best_params_)
    lgbm_model.fit(X_train, y_train) 
    
def lightGBM_regressor_pred(X_test, y_test):
    """
    Fungsi untuk membuat prediksi
    menggunakan model lightGBM regression 
    """
    # model
    lgbm_pred = lgbm_model.predict(X_test)
    # rmse
    lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    print("\033[1m" + 'RMSE dari LightGBM Regressor' + "\033[0m")
    print('RMSE: {:.3f}'.format(lgbm_rmse))
    print()
    
    # features penting dari model  
    sorted_feature_importance = lgbm_model.feature_importances_.argsort()
    plt.figure(figsize=(8,6))
    plt.barh(features_train.columns[sorted_feature_importance], 
             lgbm_model.feature_importances_[sorted_feature_importance], 
             color='turquoise')
    plt.xlabel("Feature penting dari LightGBM ")
```


```python
%%time
train_lightGBM_reg(features_train, target_train)
```

    CPU times: user 1.9 s, sys: 39.3 ms, total: 1.94 s
    Wall time: 1.92 s



```python
%%time
lightGBM_regressor_pred(features_test, target_test)
```

    [1mRMSE dari LightGBM Regressor[0m
    RMSE: 45.342
    
    CPU times: user 58.1 ms, sys: 0 ns, total: 58.1 ms
    Wall time: 100 ms



    
![png](output_87_1.png)
    


# Hasil Model 

Pada bagian ini, kita akan menganalisis kecepatan dan kualitas model yang dilatih. Ringkasan model, RMSE masing-masing, waktu yang diperlukan untuk menyetel hyperparameter, dan melatih model ditampilkan di bawah.


```python
Model_analyz = {"Model" : ["Linear Regression", "KNeighbors Regressor", "Random Forest Regressor", "Catboost Regressor", "XGBoost Regressor", "LightGBM Regressor"], "Hyperparamater Tune":['0', '3.69s', '2min 12s','12min 28s','13min 59s','1min 43s'] , "Training time" :  ['3.99 ms','9.13 ms','2.4 s','21.1 s','1.21 s','1.92 s'] , "Predict time" : ['3.41 m','13.9 ms','71.3 ms','31 ms','33.1 ms','100 ms']   , "RMSE" :  ['52.740', '62.147','44.786','47.622','45.933','45.342']}
```


```python
pd.DataFrame(Model_analyz)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Hyperparamater Tune</th>
      <th>Training time</th>
      <th>Predict time</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0</td>
      <td>3.99 ms</td>
      <td>3.41 m</td>
      <td>52.740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNeighbors Regressor</td>
      <td>3.69s</td>
      <td>9.13 ms</td>
      <td>13.9 ms</td>
      <td>62.147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest Regressor</td>
      <td>2min 12s</td>
      <td>2.4 s</td>
      <td>71.3 ms</td>
      <td>44.786</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Catboost Regressor</td>
      <td>12min 28s</td>
      <td>21.1 s</td>
      <td>31 ms</td>
      <td>47.622</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XGBoost Regressor</td>
      <td>13min 59s</td>
      <td>1.21 s</td>
      <td>33.1 ms</td>
      <td>45.933</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LightGBM Regressor</td>
      <td>1min 43s</td>
      <td>1.92 s</td>
      <td>100 ms</td>
      <td>45.342</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

Pada tahap ini kita melatih berbagai algoritme dengan berbagai hyperparameter, yakni :  Linear regressor, KNeighbors regressor, Random forest regressor, Catboost regressor, XGBoost regressor dan LightGBM regressor. kita juga menyetel berbagai hyperparameter yang memengaruhi kinerja model serta mengamati waktu yang diperlukan untuk menyetel hyperparameter, waktu latih dan waktu prediksi model. Metrik yang digunakan untuk mengevaluasi model adalah RMSE Score. Algoritma KNeighbors regressor memiliki waktu pelatihan tercepat namun memiliki skor RMSE terburuk yaitu 62,147 dan Random Forest Regressor memiliki waktu penyetelan tercepat yakni 44.786. Model dengan skor RMSE terbaik akan dipilih untuk pengujian model untuk tugas ini.

## Pengujian

Random Forest Regressor dipilih sebagai model untuk pengujian akhir untuk tugas ini karena memiliki skor RMSE terbaik. 


```python
# menguji model 
random_forest_pred(features_test, target_test)
```

    [1mRMSE dari Random Forest Regressor[0m
    RMSE: 44.786
    



    
![png](output_95_1.png)
    


Kesimpulan:
___
Menggunakan algoritma Random Forest Regressor, kita memperoleh skor RMSE: 44.786 untuk test dataset. dapat dilihat bahwa Feature penting yang di tampilkan dari algoritma Random Forest Regressor yakni :  `lag_1`, `hour`, `rolling_mean`, `month`dan `lag_2`. `dayofweek` adalah feature yang paling tidak penting untuk model ini. 


# Daftar Periksa Penilaian

- [x]  Jupyter Notebook bisa dibuka.
- [ ]  Tidak ada kesalahan dalam kode
- [ ]  Sel-sel dengan kode telah disusun berdasarkan urutan eksekusi.
- [ ]  Data telah diunduh dan disiapkan
- [ ]  Data telah dianalisis
- [ ]  Model sudah dilatih dan hiperparameter sudah dipilih
- [ ]  Model sudah dievaluasi. Kesimpulan sudah ada.
- [ ] *RMSE* untuk *test set* tidak lebih dari 48


```python

```
