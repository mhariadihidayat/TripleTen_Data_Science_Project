#  Analisis Paket Berbayar Pada Perusahaan Telekomunikasi Megaline 

Anda bekerja sebagai analis untuk operator telekomunikasi Megaline. Perusahaan tersebut menawarkan kliennya dua paket prabayar, Surf dan Ultimate. Departemen periklanan ingin mengetahui paket prabayar mana yang menghasilkan lebih banyak pendapatan untuk menyesuaikan anggaran iklan.
Anda akan melakukan analisis awal untuk paket-paket prabayar tersebut berdasarkan sampel klien yang berukuran relatif kecil. Anda akan memiliki 500 data klien Megaline: siapa mereka, dari mana mereka, jenis paket apa yang mereka gunakan, serta jumlah panggilan dan pesan yang mereka kirim di tahun 2018. Tugas Anda adalah untuk menganalisis perilaku klien dan menentukan paket prabayar mana yang mendatangkan lebih banyak pendapatan.

Deskripsi dari project ini adalah :

Sebagai seorang data scientist pada perusahaan telekomunikasi Megaline, dimana background Perusahaan ini begerak dibidang Telekomunikasi. Perusahaan ini menawarkan dua paket prabayar yaitu Surf dan Ultimate. Tujuan dari project ini adalah untuk menganalisa dua paket tersebut berdasarkan sample klien agar mengetahui paket mana yang menghasilkan pendapatan keuntungan yang lebih besar, agar departemen periklanan dapat menyesuaikan anggaran iklan. 

Langkah pada analisis ini dilakukan :

1. Pra-pemrosesan data. 
2. Data Preperation.
3. Data Analyzing.
4. Menguji Hipotesis.
5. Kesimpulan Akhir. 

## Memuat data dan mempelajari informasi keseluruhan pada data

### Memuat Libary yang dibutuhkan untuk pemrosesan data


```python
# Memuat semua library

# import pandas and numpy untuk proses dan manipulasi data
import pandas as pd
import numpy as np 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# import math dan scipy untuk perhitungan statistika 
import math as mt
from math import factorial
from scipy import stats as st

# import matplotlib untuk data visualisasi
import matplotlib.pyplot as plt 
from matplotlib import pyplot
%matplotlib inline

# Import seaborn untuk statistika data visualisasi
import seaborn as sns

# import date dan time untuk merubah tipe data
import time
import datetime
from datetime import datetime

# import warnings untuk menghapus peringatan saat dataset di manipulasi
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


```

### Memuat Data dari csv agar dapat dijalankan dengan pandas untuk menjadi DataFrame


```python
# Muat file data menjadi DataFrame

calls    = pd.read_csv('/datasets/megaline_calls.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
plans    = pd.read_csv('/datasets/megaline_plans.csv')
users    = pd.read_csv('/datasets/megaline_users.csv')

```

###  Memuat Informasi dari setiap dataset

#### Dataset calls


```python
# Informasi dataset calls

print('Head dataset "calls":')
calls.head()
```

    Head dataset "calls":





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
      <th>id</th>
      <th>user_id</th>
      <th>call_date</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_93</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>8.52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_145</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>13.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_247</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>14.48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_309</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>5.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_380</td>
      <td>1000</td>
      <td>2018-12-30</td>
      <td>4.22</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('info dataset "calls"')
calls.info()
```

    info dataset "calls"
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 137735 entries, 0 to 137734
    Data columns (total 4 columns):
     #   Column     Non-Null Count   Dtype  
    ---  ------     --------------   -----  
     0   id         137735 non-null  object 
     1   user_id    137735 non-null  int64  
     2   call_date  137735 non-null  object 
     3   duration   137735 non-null  float64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 4.2+ MB



```python
print('distribusi statisik pada dataser "calls"')
calls.describe()
```

    distribusi statisik pada dataser "calls"





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
      <th>user_id</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>137735.000000</td>
      <td>137735.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1247.658046</td>
      <td>6.745927</td>
    </tr>
    <tr>
      <th>std</th>
      <td>139.416268</td>
      <td>5.839241</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1128.000000</td>
      <td>1.290000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1247.000000</td>
      <td>5.980000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1365.000000</td>
      <td>10.690000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1499.000000</td>
      <td>37.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na')
calls.isna().sum()
```

    memeriksa nilai yang hilang / na





    id           0
    user_id      0
    call_date    0
    duration     0
    dtype: int64




```python
print('panjang baris dari dataset "calls":')
calls.shape
```

    panjang baris dari dataset "calls":





    (137735, 4)




```python
print('memeriksa duplikasi pada dataset "calls":')
print('nilai duplikat pada dataset "calls":', calls.duplicated().sum())
```

    memeriksa duplikasi pada dataset "calls":
    nilai duplikat pada dataset "calls": 0


#### Dataset Internet


```python
print('Head dataset "internet":')
internet.head()
```

    Head dataset "internet":





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
      <th>id</th>
      <th>user_id</th>
      <th>session_date</th>
      <th>mb_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_13</td>
      <td>1000</td>
      <td>2018-12-29</td>
      <td>89.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_204</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_379</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>660.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_413</td>
      <td>1000</td>
      <td>2018-12-26</td>
      <td>270.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_442</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>880.22</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('info dataset "internet"')
internet.info()
```

    info dataset "internet"
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 104825 entries, 0 to 104824
    Data columns (total 4 columns):
     #   Column        Non-Null Count   Dtype  
    ---  ------        --------------   -----  
     0   id            104825 non-null  object 
     1   user_id       104825 non-null  int64  
     2   session_date  104825 non-null  object 
     3   mb_used       104825 non-null  float64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 3.2+ MB



```python
print('distribusi statisik pada dataser "internet"')
internet.describe()
```

    distribusi statisik pada dataser "internet"





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
      <th>user_id</th>
      <th>mb_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>104825.000000</td>
      <td>104825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1242.496361</td>
      <td>366.713701</td>
    </tr>
    <tr>
      <th>std</th>
      <td>142.053913</td>
      <td>277.170542</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1122.000000</td>
      <td>136.080000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1236.000000</td>
      <td>343.980000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1367.000000</td>
      <td>554.610000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1499.000000</td>
      <td>1693.470000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na')
internet.isna().sum()
```

    memeriksa nilai yang hilang / na





    id              0
    user_id         0
    session_date    0
    mb_used         0
    dtype: int64




```python
print('panjang baris dari dataset "internet":')
internet.shape
```

    panjang baris dari dataset "internet":





    (104825, 4)




```python
print('memeriksa duplikasi pada dataset "internet":')
print('nilai duplikat pada dataset "internet":', internet.duplicated().sum())
```

    memeriksa duplikasi pada dataset "internet":
    nilai duplikat pada dataset "internet": 0


#### Dataset Messages


```python
print('Head dataset "messages":')
messages.head()
```

    Head dataset "messages":





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
      <th>id</th>
      <th>user_id</th>
      <th>message_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_125</td>
      <td>1000</td>
      <td>2018-12-27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_160</td>
      <td>1000</td>
      <td>2018-12-31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_223</td>
      <td>1000</td>
      <td>2018-12-31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_251</td>
      <td>1000</td>
      <td>2018-12-27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_255</td>
      <td>1000</td>
      <td>2018-12-26</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('info dataset "messages"')
messages.info()
```

    info dataset "messages"
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 76051 entries, 0 to 76050
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   id            76051 non-null  object
     1   user_id       76051 non-null  int64 
     2   message_date  76051 non-null  object
    dtypes: int64(1), object(2)
    memory usage: 1.7+ MB



```python
print('distribusi statisik pada dataser "messages"')
messages.describe()
```

    distribusi statisik pada dataser "messages"





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
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>76051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1245.972768</td>
    </tr>
    <tr>
      <th>std</th>
      <td>139.843635</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1123.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1251.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1362.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1497.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na')
messages.isna().sum()
```

    memeriksa nilai yang hilang / na





    id              0
    user_id         0
    message_date    0
    dtype: int64




```python
print('panjang baris dari dataset "messages":')
messages.shape
```

    panjang baris dari dataset "messages":





    (76051, 3)




```python
print('memeriksa duplikasi pada dataset "messages":')
print('nilai duplikat pada dataset "messages":', messages.duplicated().sum())
```

    memeriksa duplikasi pada dataset "messages":
    nilai duplikat pada dataset "messages": 0


#### Dataset Plans


```python
print('Head dataset "plans":')
plans.head()
```

    Head dataset "plans":





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
      <th>messages_included</th>
      <th>mb_per_month_included</th>
      <th>minutes_included</th>
      <th>usd_monthly_pay</th>
      <th>usd_per_gb</th>
      <th>usd_per_message</th>
      <th>usd_per_minute</th>
      <th>plan_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>surf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>ultimate</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('info dataset "plans"')
plans.info()
```

    info dataset "plans"
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2 entries, 0 to 1
    Data columns (total 8 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   messages_included      2 non-null      int64  
     1   mb_per_month_included  2 non-null      int64  
     2   minutes_included       2 non-null      int64  
     3   usd_monthly_pay        2 non-null      int64  
     4   usd_per_gb             2 non-null      int64  
     5   usd_per_message        2 non-null      float64
     6   usd_per_minute         2 non-null      float64
     7   plan_name              2 non-null      object 
    dtypes: float64(2), int64(5), object(1)
    memory usage: 256.0+ bytes



```python
print('distribusi statisik pada dataser "plans"')
plans.describe()
```

    distribusi statisik pada dataser "plans"





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
      <th>messages_included</th>
      <th>mb_per_month_included</th>
      <th>minutes_included</th>
      <th>usd_monthly_pay</th>
      <th>usd_per_gb</th>
      <th>usd_per_message</th>
      <th>usd_per_minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.00000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>525.000000</td>
      <td>23040.000000</td>
      <td>1750.000000</td>
      <td>45.000000</td>
      <td>8.50000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>671.751442</td>
      <td>10861.160159</td>
      <td>1767.766953</td>
      <td>35.355339</td>
      <td>2.12132</td>
      <td>0.014142</td>
      <td>0.014142</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.000000</td>
      <td>15360.000000</td>
      <td>500.000000</td>
      <td>20.000000</td>
      <td>7.00000</td>
      <td>0.010000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>287.500000</td>
      <td>19200.000000</td>
      <td>1125.000000</td>
      <td>32.500000</td>
      <td>7.75000</td>
      <td>0.015000</td>
      <td>0.015000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>525.000000</td>
      <td>23040.000000</td>
      <td>1750.000000</td>
      <td>45.000000</td>
      <td>8.50000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>762.500000</td>
      <td>26880.000000</td>
      <td>2375.000000</td>
      <td>57.500000</td>
      <td>9.25000</td>
      <td>0.025000</td>
      <td>0.025000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1000.000000</td>
      <td>30720.000000</td>
      <td>3000.000000</td>
      <td>70.000000</td>
      <td>10.00000</td>
      <td>0.030000</td>
      <td>0.030000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(' ID dari dataset plans: ')
plans['plan_name'].describe().reset_index()
```

     ID dari dataset plans: 





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
      <th>index</th>
      <th>plan_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>unique</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>top</td>
      <td>surf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>freq</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na')
plans.isna().sum()
```

    memeriksa nilai yang hilang / na





    messages_included        0
    mb_per_month_included    0
    minutes_included         0
    usd_monthly_pay          0
    usd_per_gb               0
    usd_per_message          0
    usd_per_minute           0
    plan_name                0
    dtype: int64




```python
print('panjang baris dari dataset "plans":')
plans.shape
```

    panjang baris dari dataset "plans":





    (2, 8)




```python
print('memeriksa duplikasi pada dataset "plans":')
print('nilai duplikat pada dataset "plans":', plans.duplicated().sum())
```

    memeriksa duplikasi pada dataset "plans":
    nilai duplikat pada dataset "plans": 0


#### Dataset Users


```python
print('Head dataset "users":')
users.head()
```

    Head dataset "users":





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
      <th>user_id</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>reg_date</th>
      <th>plan</th>
      <th>churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>Atlanta-Sandy Springs-Roswell, GA MSA</td>
      <td>2018-12-24</td>
      <td>ultimate</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>Carlee</td>
      <td>Hoffman</td>
      <td>36</td>
      <td>Las Vegas-Henderson-Paradise, NV MSA</td>
      <td>2018-10-21</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>Reynaldo</td>
      <td>Jenkins</td>
      <td>52</td>
      <td>Tulsa, OK MSA</td>
      <td>2018-01-28</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>Leonila</td>
      <td>Thompson</td>
      <td>40</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-05-23</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('info dataset "users"')
users.info()
```

    info dataset "users"
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   user_id     500 non-null    int64 
     1   first_name  500 non-null    object
     2   last_name   500 non-null    object
     3   age         500 non-null    int64 
     4   city        500 non-null    object
     5   reg_date    500 non-null    object
     6   plan        500 non-null    object
     7   churn_date  34 non-null     object
    dtypes: int64(2), object(6)
    memory usage: 31.4+ KB



```python
print('distribusi statisik pada dataset "users"')
users.describe()
```

    distribusi statisik pada dataset "users"





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
      <th>user_id</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1249.500000</td>
      <td>45.486000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>144.481833</td>
      <td>16.972269</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1124.750000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1249.500000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1374.250000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1499.000000</td>
      <td>75.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(' ID dari dataset plans: ')
users[['first_name', 'last_name','city', 'reg_date', 'plan', 'churn_date']].describe().reset_index()
```

     ID dari dataset plans: 





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
      <th>index</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>city</th>
      <th>reg_date</th>
      <th>plan</th>
      <th>churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>500</td>
      <td>500</td>
      <td>500</td>
      <td>500</td>
      <td>500</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>unique</td>
      <td>458</td>
      <td>399</td>
      <td>73</td>
      <td>266</td>
      <td>2</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>top</td>
      <td>Seymour</td>
      <td>David</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-07-12</td>
      <td>surf</td>
      <td>2018-12-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>freq</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>5</td>
      <td>339</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na')
users.isna().sum()
```

    memeriksa nilai yang hilang / na





    user_id         0
    first_name      0
    last_name       0
    age             0
    city            0
    reg_date        0
    plan            0
    churn_date    466
    dtype: int64




```python
print('Menghitung nilai yang hilang pada column churn_date')
percent_missing = users['churn_date'].isnull().sum() * 100 / len(users)
print(f'persentase jumlah nilai yang hilang pada column churn_date : {percent_missing} %') 
```

    Menghitung nilai yang hilang pada column churn_date
    persentase jumlah nilai yang hilang pada column churn_date : 93.2 %



```python
print('panjang baris dari dataset "users":')
users.shape
```

    panjang baris dari dataset "users":





    (500, 8)




```python
print('memeriksa duplikasi pada dataset "users":')
print('nilai duplikat pada dataset "users":', users.duplicated().sum())
```

    memeriksa duplikasi pada dataset "users":
    nilai duplikat pada dataset "users": 0


#### Kesimpulan :

1. Dari informasi keseluruhan dari dataset, kita dapat mengidentifikasi beberapa dataset terdapat beberapa column yang memliki tipe data yang salah. untuk itu kita perlu merubah tipe data tersebut.
_______________________________________________________________________________________________________________________________

2. column yang perlu diubah tipe datanya :
a. call_date pada dataset calls perlu diubah menjadi tipe data datetime.
b. session_date pada dataset internet perlu diubah datanya menjadi tipe data datetime.
c. message_date pada dataset message perlu diubah datanya menjadi tipe data datetime.
d. reg_date dan churn_date pada dataset users perlu diubah menjadi tipe data datetime.
_______________________________________________________________________________________________________________________________

3. selanjutnya Kita juga perlu mengidentifikasi lebih detail mengenai pola pada nilai yang hilang pada column users. dan mengganti tipe data yang dibutuhkan

## Data Preperation

#### Data Calls Preparation



```python
# Fungsi untuk mengubah date type to datetime dan memisahkannya ke day, month and year
def new_date_features(df):
    columns = df.columns.tolist()
    idx = [columns.index(x) for x in columns if 'date' in x][0]
    
    df[columns[idx]] = pd.to_datetime(df[columns[idx]])
    df['day'] = df[columns[idx]].dt.day_name()
    df['month'] = df[columns[idx]].dt.month_name()
    df['year'] = df[columns[idx]].dt.year
    return df    
```


```python
# Merubah column calls_date menjadi tipe data datetime 

calls['call_date'] = pd.to_datetime(calls['call_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
calls.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 137735 entries, 0 to 137734
    Data columns (total 4 columns):
     #   Column     Non-Null Count   Dtype         
    ---  ------     --------------   -----         
     0   id         137735 non-null  object        
     1   user_id    137735 non-null  int64         
     2   call_date  137735 non-null  datetime64[ns]
     3   duration   137735 non-null  float64       
    dtypes: datetime64[ns](1), float64(1), int64(1), object(1)
    memory usage: 4.2+ MB



```python
# Menerapkan fungsi untuk menambahkan day, month and year

calls = new_date_features(calls)
calls.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>call_date</th>
      <th>duration</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_93</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>8.52</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_145</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>13.66</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_247</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>14.48</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_309</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>5.76</td>
      <td>Friday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_380</td>
      <td>1000</td>
      <td>2018-12-30</td>
      <td>4.22</td>
      <td>Sunday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menambahkan column call_type untuk mengidentifikasi panggilan tak terjawab

calls['call_type'] = np.where(calls['duration'] == 0, 'Missed_call', 'Connected_call')
calls.groupby('call_type')['duration'].agg('value_counts')
```




    call_type       duration
    Connected_call  4.02          102
                    8.37          102
                    3.91          101
                    4.30          100
                    7.61          100
                                ...  
                    35.74           1
                    35.88           1
                    36.24           1
                    37.60           1
    Missed_call     0.00        26834
    Name: duration, Length: 2802, dtype: int64




```python
# Merubah column duration menjadi tipedata interger untuk membulatkan value 

calls['duration'] = (calls['duration'].apply(np.ceil).astype('int64'))
```


```python
# Merubah nama column

calls.columns = ['id', 'user', 'call_date', 'duration', 'day', 'month', 'year', 'call_type']
calls.head() 
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
      <th>id</th>
      <th>user</th>
      <th>call_date</th>
      <th>duration</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
      <th>call_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_93</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>9</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
      <td>Connected_call</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_145</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>14</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
      <td>Connected_call</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_247</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>15</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
      <td>Connected_call</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_309</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>6</td>
      <td>Friday</td>
      <td>December</td>
      <td>2018</td>
      <td>Connected_call</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_380</td>
      <td>1000</td>
      <td>2018-12-30</td>
      <td>5</td>
      <td>Sunday</td>
      <td>December</td>
      <td>2018</td>
      <td>Connected_call</td>
    </tr>
  </tbody>
</table>
</div>



#### Data Internet Preparation 


```python
internet.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>session_date</th>
      <th>mb_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_13</td>
      <td>1000</td>
      <td>2018-12-29</td>
      <td>89.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_204</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_379</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>660.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_413</td>
      <td>1000</td>
      <td>2018-12-26</td>
      <td>270.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_442</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>880.22</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merubah tipe data mb_used ke bilangan bulat
internet['mb_used'] = internet['mb_used'].astype('int')
```


```python
# Merubah column session date menjadi tipe data datetime 

internet['session_date'] = pd.to_datetime(internet['session_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
internet.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 104825 entries, 0 to 104824
    Data columns (total 4 columns):
     #   Column        Non-Null Count   Dtype         
    ---  ------        --------------   -----         
     0   id            104825 non-null  object        
     1   user_id       104825 non-null  int64         
     2   session_date  104825 non-null  datetime64[ns]
     3   mb_used       104825 non-null  int64         
    dtypes: datetime64[ns](1), int64(2), object(1)
    memory usage: 3.2+ MB



```python
# Menerapkan fungsi untuk menambahkan day, month and year

internet = new_date_features(internet)
internet
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
      <th>id</th>
      <th>user_id</th>
      <th>session_date</th>
      <th>mb_used</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_13</td>
      <td>1000</td>
      <td>2018-12-29</td>
      <td>89</td>
      <td>Saturday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_204</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>0</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_379</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>660</td>
      <td>Friday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_413</td>
      <td>1000</td>
      <td>2018-12-26</td>
      <td>270</td>
      <td>Wednesday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_442</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>880</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>104820</th>
      <td>1499_215</td>
      <td>1499</td>
      <td>2018-10-20</td>
      <td>218</td>
      <td>Saturday</td>
      <td>October</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104821</th>
      <td>1499_216</td>
      <td>1499</td>
      <td>2018-12-30</td>
      <td>304</td>
      <td>Sunday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104822</th>
      <td>1499_217</td>
      <td>1499</td>
      <td>2018-09-22</td>
      <td>292</td>
      <td>Saturday</td>
      <td>September</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104823</th>
      <td>1499_218</td>
      <td>1499</td>
      <td>2018-12-07</td>
      <td>0</td>
      <td>Friday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104824</th>
      <td>1499_219</td>
      <td>1499</td>
      <td>2018-12-24</td>
      <td>758</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>104825 rows × 7 columns</p>
</div>




```python
# Merubah nama column

internet.columns = ['id', 'user', 'session_date', 'data_used', 'day', 'month', 'year']
internet
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
      <th>id</th>
      <th>user</th>
      <th>session_date</th>
      <th>data_used</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_13</td>
      <td>1000</td>
      <td>2018-12-29</td>
      <td>89</td>
      <td>Saturday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_204</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>0</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_379</td>
      <td>1000</td>
      <td>2018-12-28</td>
      <td>660</td>
      <td>Friday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_413</td>
      <td>1000</td>
      <td>2018-12-26</td>
      <td>270</td>
      <td>Wednesday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_442</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>880</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>104820</th>
      <td>1499_215</td>
      <td>1499</td>
      <td>2018-10-20</td>
      <td>218</td>
      <td>Saturday</td>
      <td>October</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104821</th>
      <td>1499_216</td>
      <td>1499</td>
      <td>2018-12-30</td>
      <td>304</td>
      <td>Sunday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104822</th>
      <td>1499_217</td>
      <td>1499</td>
      <td>2018-09-22</td>
      <td>292</td>
      <td>Saturday</td>
      <td>September</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104823</th>
      <td>1499_218</td>
      <td>1499</td>
      <td>2018-12-07</td>
      <td>0</td>
      <td>Friday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>104824</th>
      <td>1499_219</td>
      <td>1499</td>
      <td>2018-12-24</td>
      <td>758</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>104825 rows × 7 columns</p>
</div>



#### Data Messages Preparation


```python
messages.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>message_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_125</td>
      <td>1000</td>
      <td>2018-12-27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_160</td>
      <td>1000</td>
      <td>2018-12-31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_223</td>
      <td>1000</td>
      <td>2018-12-31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_251</td>
      <td>1000</td>
      <td>2018-12-27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_255</td>
      <td>1000</td>
      <td>2018-12-26</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merubah column message_date menjadi tipe data datetime 

messages['message_date'] = pd.to_datetime(messages['message_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
```


```python
# Menerapkan fungsi untuk menambahkan day, month and year

messages = new_date_features(messages)
messages.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>message_date</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_125</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_160</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_223</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_251</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_255</td>
      <td>1000</td>
      <td>2018-12-26</td>
      <td>Wednesday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merubah nama column

messages.columns = ['id', 'user', 'message date', 'day', 'month', 'year']
messages.head()
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
      <th>id</th>
      <th>user</th>
      <th>message date</th>
      <th>day</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000_125</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000_160</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000_223</td>
      <td>1000</td>
      <td>2018-12-31</td>
      <td>Monday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000_251</td>
      <td>1000</td>
      <td>2018-12-27</td>
      <td>Thursday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000_255</td>
      <td>1000</td>
      <td>2018-12-26</td>
      <td>Wednesday</td>
      <td>December</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>



#### Data Plans Preparation


```python
plans.head()
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
      <th>messages_included</th>
      <th>mb_per_month_included</th>
      <th>minutes_included</th>
      <th>usd_monthly_pay</th>
      <th>usd_per_gb</th>
      <th>usd_per_message</th>
      <th>usd_per_minute</th>
      <th>plan_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>surf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>ultimate</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merubah nama column

plans.columns = ['messages_included', 'data_volume_per_month', 'minutes_included', 'monthly_fee', 'price_per_gb', 'price_per_message', 'price_per_minute', 'plan']
plans.head()
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
      <th>messages_included</th>
      <th>data_volume_per_month</th>
      <th>minutes_included</th>
      <th>monthly_fee</th>
      <th>price_per_gb</th>
      <th>price_per_message</th>
      <th>price_per_minute</th>
      <th>plan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>surf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>ultimate</td>
    </tr>
  </tbody>
</table>
</div>



#### Data Users Preparation


```python
users.head(25)
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
      <th>user_id</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>reg_date</th>
      <th>plan</th>
      <th>churn_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>Atlanta-Sandy Springs-Roswell, GA MSA</td>
      <td>2018-12-24</td>
      <td>ultimate</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>Carlee</td>
      <td>Hoffman</td>
      <td>36</td>
      <td>Las Vegas-Henderson-Paradise, NV MSA</td>
      <td>2018-10-21</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>Reynaldo</td>
      <td>Jenkins</td>
      <td>52</td>
      <td>Tulsa, OK MSA</td>
      <td>2018-01-28</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>Leonila</td>
      <td>Thompson</td>
      <td>40</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-05-23</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1005</td>
      <td>Livia</td>
      <td>Shields</td>
      <td>31</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-11-29</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1006</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>San Francisco-Oakland-Berkeley, CA MSA</td>
      <td>2018-11-27</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1007</td>
      <td>Eusebio</td>
      <td>Welch</td>
      <td>42</td>
      <td>Grand Rapids-Kentwood, MI MSA</td>
      <td>2018-07-11</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1008</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-08-03</td>
      <td>ultimate</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1009</td>
      <td>Gerry</td>
      <td>Little</td>
      <td>19</td>
      <td>San Jose-Sunnyvale-Santa Clara, CA MSA</td>
      <td>2018-04-22</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1010</td>
      <td>Wilber</td>
      <td>Blair</td>
      <td>52</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-03-09</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1011</td>
      <td>Halina</td>
      <td>Henry</td>
      <td>73</td>
      <td>Cleveland-Elyria, OH MSA</td>
      <td>2018-01-18</td>
      <td>ultimate</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1012</td>
      <td>Jonelle</td>
      <td>Mcbride</td>
      <td>59</td>
      <td>Chicago-Naperville-Elgin, IL-IN-WI MSA</td>
      <td>2018-06-28</td>
      <td>surf</td>
      <td>2018-11-16</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1013</td>
      <td>Nicolas</td>
      <td>Snider</td>
      <td>50</td>
      <td>Knoxville, TN MSA</td>
      <td>2018-12-01</td>
      <td>ultimate</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1014</td>
      <td>Edmundo</td>
      <td>Simon</td>
      <td>61</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-11-25</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1015</td>
      <td>Beata</td>
      <td>Carpenter</td>
      <td>26</td>
      <td>Pittsburgh, PA MSA</td>
      <td>2018-12-05</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1016</td>
      <td>Jann</td>
      <td>Salinas</td>
      <td>30</td>
      <td>Fresno, CA MSA</td>
      <td>2018-10-25</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1017</td>
      <td>Boris</td>
      <td>Gates</td>
      <td>61</td>
      <td>Washington-Arlington-Alexandria, DC-VA-MD-WV MSA</td>
      <td>2018-08-26</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1018</td>
      <td>Dennis</td>
      <td>Grimes</td>
      <td>70</td>
      <td>Indianapolis-Carmel-Anderson, IN MSA</td>
      <td>2018-10-17</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1019</td>
      <td>Shizue</td>
      <td>Landry</td>
      <td>34</td>
      <td>Jacksonville, FL MSA</td>
      <td>2018-01-16</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1020</td>
      <td>Rutha</td>
      <td>Bell</td>
      <td>56</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-11-08</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1021</td>
      <td>Ricarda</td>
      <td>Booker</td>
      <td>37</td>
      <td>Los Angeles-Long Beach-Anaheim, CA MSA</td>
      <td>2018-12-21</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1022</td>
      <td>Bo</td>
      <td>Snow</td>
      <td>73</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-04-20</td>
      <td>surf</td>
      <td>2018-09-07</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1023</td>
      <td>Jack</td>
      <td>Delaney</td>
      <td>70</td>
      <td>Omaha-Council Bluffs, NE-IA MSA</td>
      <td>2018-07-06</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1024</td>
      <td>Yuki</td>
      <td>Tyson</td>
      <td>74</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-08-21</td>
      <td>surf</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merubah column message_date menjadi tipe data datetime 

users['reg_date'] = pd.to_datetime(users['reg_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
```


```python
# Merubah column churn_date menjadi tipe data datetime 

users['churn_date'] = pd.to_datetime(users['churn_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
```


```python
# Meynyesuaikan nama column

users.columns = ['user', 'first_name', 'last_name', 'age', 'city', 'subscription_date', 'plan', 'churn_date']
```


```python
# membuat kategori untuk nilai yang hilang pada column customer_churn berdasarkan churn_date

users['customer_churn'] = np.where(users['churn_date'].isnull(), 'No', 'Yes')
users.head(25) 
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
      <th>user</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>subscription_date</th>
      <th>plan</th>
      <th>churn_date</th>
      <th>customer_churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>Atlanta-Sandy Springs-Roswell, GA MSA</td>
      <td>2018-12-24</td>
      <td>ultimate</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>Carlee</td>
      <td>Hoffman</td>
      <td>36</td>
      <td>Las Vegas-Henderson-Paradise, NV MSA</td>
      <td>2018-10-21</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>Reynaldo</td>
      <td>Jenkins</td>
      <td>52</td>
      <td>Tulsa, OK MSA</td>
      <td>2018-01-28</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>Leonila</td>
      <td>Thompson</td>
      <td>40</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-05-23</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1005</td>
      <td>Livia</td>
      <td>Shields</td>
      <td>31</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-11-29</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1006</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>San Francisco-Oakland-Berkeley, CA MSA</td>
      <td>2018-11-27</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1007</td>
      <td>Eusebio</td>
      <td>Welch</td>
      <td>42</td>
      <td>Grand Rapids-Kentwood, MI MSA</td>
      <td>2018-07-11</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1008</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-08-03</td>
      <td>ultimate</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1009</td>
      <td>Gerry</td>
      <td>Little</td>
      <td>19</td>
      <td>San Jose-Sunnyvale-Santa Clara, CA MSA</td>
      <td>2018-04-22</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1010</td>
      <td>Wilber</td>
      <td>Blair</td>
      <td>52</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-03-09</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1011</td>
      <td>Halina</td>
      <td>Henry</td>
      <td>73</td>
      <td>Cleveland-Elyria, OH MSA</td>
      <td>2018-01-18</td>
      <td>ultimate</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1012</td>
      <td>Jonelle</td>
      <td>Mcbride</td>
      <td>59</td>
      <td>Chicago-Naperville-Elgin, IL-IN-WI MSA</td>
      <td>2018-06-28</td>
      <td>surf</td>
      <td>2018-11-16</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1013</td>
      <td>Nicolas</td>
      <td>Snider</td>
      <td>50</td>
      <td>Knoxville, TN MSA</td>
      <td>2018-12-01</td>
      <td>ultimate</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1014</td>
      <td>Edmundo</td>
      <td>Simon</td>
      <td>61</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-11-25</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1015</td>
      <td>Beata</td>
      <td>Carpenter</td>
      <td>26</td>
      <td>Pittsburgh, PA MSA</td>
      <td>2018-12-05</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1016</td>
      <td>Jann</td>
      <td>Salinas</td>
      <td>30</td>
      <td>Fresno, CA MSA</td>
      <td>2018-10-25</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1017</td>
      <td>Boris</td>
      <td>Gates</td>
      <td>61</td>
      <td>Washington-Arlington-Alexandria, DC-VA-MD-WV MSA</td>
      <td>2018-08-26</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1018</td>
      <td>Dennis</td>
      <td>Grimes</td>
      <td>70</td>
      <td>Indianapolis-Carmel-Anderson, IN MSA</td>
      <td>2018-10-17</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1019</td>
      <td>Shizue</td>
      <td>Landry</td>
      <td>34</td>
      <td>Jacksonville, FL MSA</td>
      <td>2018-01-16</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1020</td>
      <td>Rutha</td>
      <td>Bell</td>
      <td>56</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-11-08</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1021</td>
      <td>Ricarda</td>
      <td>Booker</td>
      <td>37</td>
      <td>Los Angeles-Long Beach-Anaheim, CA MSA</td>
      <td>2018-12-21</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1022</td>
      <td>Bo</td>
      <td>Snow</td>
      <td>73</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-04-20</td>
      <td>surf</td>
      <td>2018-09-07</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1023</td>
      <td>Jack</td>
      <td>Delaney</td>
      <td>70</td>
      <td>Omaha-Council Bluffs, NE-IA MSA</td>
      <td>2018-07-06</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1024</td>
      <td>Yuki</td>
      <td>Tyson</td>
      <td>74</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-08-21</td>
      <td>surf</td>
      <td>NaT</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
# menampilkan jumlah nilai yang hilang (NaT) pada column churn_date

users['churn_date'].isnull().sum()
```




    466




```python
# Menampilkan nilai unique pada column churn_date

users['churn_date'].sort_values().unique()
```




    array(['2018-07-31T00:00:00.000000000', '2018-08-16T00:00:00.000000000',
           '2018-08-19T00:00:00.000000000', '2018-09-01T00:00:00.000000000',
           '2018-09-07T00:00:00.000000000', '2018-09-17T00:00:00.000000000',
           '2018-09-18T00:00:00.000000000', '2018-10-03T00:00:00.000000000',
           '2018-10-07T00:00:00.000000000', '2018-10-13T00:00:00.000000000',
           '2018-10-22T00:00:00.000000000', '2018-11-11T00:00:00.000000000',
           '2018-11-14T00:00:00.000000000', '2018-11-16T00:00:00.000000000',
           '2018-11-18T00:00:00.000000000', '2018-11-21T00:00:00.000000000',
           '2018-11-24T00:00:00.000000000', '2018-11-29T00:00:00.000000000',
           '2018-11-30T00:00:00.000000000', '2018-12-10T00:00:00.000000000',
           '2018-12-12T00:00:00.000000000', '2018-12-15T00:00:00.000000000',
           '2018-12-18T00:00:00.000000000', '2018-12-19T00:00:00.000000000',
           '2018-12-22T00:00:00.000000000', '2018-12-26T00:00:00.000000000',
           '2018-12-27T00:00:00.000000000', '2018-12-30T00:00:00.000000000',
           '2018-12-31T00:00:00.000000000',                           'NaT'],
          dtype='datetime64[ns]')




```python
# Mengisi nilai yang hilang NaT (missing value datetime type) mengisi dengan tanggal dimana data ini dibuat.  

u = users.select_dtypes(include=['datetime'])
users[u.columns] = u.fillna(pd.to_datetime('2018-01-01'))

```


```python
# menampilkan jumlah nilai yang hilang (NaT) pada column churn_date

users['churn_date'].isnull().sum()
```




    0




```python
# Menampilkan nilai unique pada column churn_date

users['churn_date'].sort_values().unique()
```




    array(['2018-01-01T00:00:00.000000000', '2018-07-31T00:00:00.000000000',
           '2018-08-16T00:00:00.000000000', '2018-08-19T00:00:00.000000000',
           '2018-09-01T00:00:00.000000000', '2018-09-07T00:00:00.000000000',
           '2018-09-17T00:00:00.000000000', '2018-09-18T00:00:00.000000000',
           '2018-10-03T00:00:00.000000000', '2018-10-07T00:00:00.000000000',
           '2018-10-13T00:00:00.000000000', '2018-10-22T00:00:00.000000000',
           '2018-11-11T00:00:00.000000000', '2018-11-14T00:00:00.000000000',
           '2018-11-16T00:00:00.000000000', '2018-11-18T00:00:00.000000000',
           '2018-11-21T00:00:00.000000000', '2018-11-24T00:00:00.000000000',
           '2018-11-29T00:00:00.000000000', '2018-11-30T00:00:00.000000000',
           '2018-12-10T00:00:00.000000000', '2018-12-12T00:00:00.000000000',
           '2018-12-15T00:00:00.000000000', '2018-12-18T00:00:00.000000000',
           '2018-12-19T00:00:00.000000000', '2018-12-22T00:00:00.000000000',
           '2018-12-26T00:00:00.000000000', '2018-12-27T00:00:00.000000000',
           '2018-12-30T00:00:00.000000000', '2018-12-31T00:00:00.000000000'],
          dtype='datetime64[ns]')




```python
users.head(20)
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
      <th>user</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>subscription_date</th>
      <th>plan</th>
      <th>churn_date</th>
      <th>customer_churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>Atlanta-Sandy Springs-Roswell, GA MSA</td>
      <td>2018-12-24</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>Carlee</td>
      <td>Hoffman</td>
      <td>36</td>
      <td>Las Vegas-Henderson-Paradise, NV MSA</td>
      <td>2018-10-21</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>Reynaldo</td>
      <td>Jenkins</td>
      <td>52</td>
      <td>Tulsa, OK MSA</td>
      <td>2018-01-28</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>Leonila</td>
      <td>Thompson</td>
      <td>40</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-05-23</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1005</td>
      <td>Livia</td>
      <td>Shields</td>
      <td>31</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-11-29</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1006</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>San Francisco-Oakland-Berkeley, CA MSA</td>
      <td>2018-11-27</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1007</td>
      <td>Eusebio</td>
      <td>Welch</td>
      <td>42</td>
      <td>Grand Rapids-Kentwood, MI MSA</td>
      <td>2018-07-11</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1008</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-08-03</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1009</td>
      <td>Gerry</td>
      <td>Little</td>
      <td>19</td>
      <td>San Jose-Sunnyvale-Santa Clara, CA MSA</td>
      <td>2018-04-22</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1010</td>
      <td>Wilber</td>
      <td>Blair</td>
      <td>52</td>
      <td>Dallas-Fort Worth-Arlington, TX MSA</td>
      <td>2018-03-09</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1011</td>
      <td>Halina</td>
      <td>Henry</td>
      <td>73</td>
      <td>Cleveland-Elyria, OH MSA</td>
      <td>2018-01-18</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1012</td>
      <td>Jonelle</td>
      <td>Mcbride</td>
      <td>59</td>
      <td>Chicago-Naperville-Elgin, IL-IN-WI MSA</td>
      <td>2018-06-28</td>
      <td>surf</td>
      <td>2018-11-16</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1013</td>
      <td>Nicolas</td>
      <td>Snider</td>
      <td>50</td>
      <td>Knoxville, TN MSA</td>
      <td>2018-12-01</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1014</td>
      <td>Edmundo</td>
      <td>Simon</td>
      <td>61</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-11-25</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1015</td>
      <td>Beata</td>
      <td>Carpenter</td>
      <td>26</td>
      <td>Pittsburgh, PA MSA</td>
      <td>2018-12-05</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1016</td>
      <td>Jann</td>
      <td>Salinas</td>
      <td>30</td>
      <td>Fresno, CA MSA</td>
      <td>2018-10-25</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1017</td>
      <td>Boris</td>
      <td>Gates</td>
      <td>61</td>
      <td>Washington-Arlington-Alexandria, DC-VA-MD-WV MSA</td>
      <td>2018-08-26</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1018</td>
      <td>Dennis</td>
      <td>Grimes</td>
      <td>70</td>
      <td>Indianapolis-Carmel-Anderson, IN MSA</td>
      <td>2018-10-17</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1019</td>
      <td>Shizue</td>
      <td>Landry</td>
      <td>34</td>
      <td>Jacksonville, FL MSA</td>
      <td>2018-01-16</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan : 

1. Setelah mengidentifikasi dataset pada kesimpulan sebelumnya, pada kesimpulan kali ini kita memperbaiki beberapa permasalahan yang telah di identifikasi. salah satunya ada pada dataset users pada column subscription_date & churn_date, lalu call_date pada calls dataset, message_date pada dataset message dan session_date pada dataset internet dimana tipe data diatas adalaha object dan kita merubah tipedata tersebut ke tipe data datetime.
_______________________________________

2. Pada nilai yang hilang NaT kita telah mengisi value nya dengan 2018-01-01
_______________________________________

3. Pada dataset users kita juga menambahkan column costumer_churn untuk kita mendefenisikan bahwa costumer yang tidak memperpanjang paket internet dengan nilai kategorik 'no' dan costumer yang memperpanjang paket internet dengan nilai kategori 'yes'

#### Meghitung Panggilan yang dilakukan, dan menit yang digunakan perbulan


```python
# Menghitung Jumlah panggilan yang dilakukan perbulan 

calls_per_month = calls.groupby(['user', 'month', 'year']).agg({'id': 'count'}).rename(columns={'id': 'calls made'})
calls_per_month.head(20)
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
      <th></th>
      <th></th>
      <th>calls made</th>
    </tr>
    <tr>
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <th>December</th>
      <th>2018</th>
      <td>16</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1001</th>
      <th>August</th>
      <th>2018</th>
      <td>27</td>
    </tr>
    <tr>
      <th>December</th>
      <th>2018</th>
      <td>56</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>64</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>65</td>
    </tr>
    <tr>
      <th>September</th>
      <th>2018</th>
      <td>49</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1002</th>
      <th>December</th>
      <th>2018</th>
      <td>47</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>55</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>11</td>
    </tr>
    <tr>
      <th>1003</th>
      <th>December</th>
      <th>2018</th>
      <td>149</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">1004</th>
      <th>August</th>
      <th>2018</th>
      <td>49</td>
    </tr>
    <tr>
      <th>December</th>
      <th>2018</th>
      <td>50</td>
    </tr>
    <tr>
      <th>July</th>
      <th>2018</th>
      <td>49</td>
    </tr>
    <tr>
      <th>June</th>
      <th>2018</th>
      <td>44</td>
    </tr>
    <tr>
      <th>May</th>
      <th>2018</th>
      <td>21</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>54</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>61</td>
    </tr>
    <tr>
      <th>September</th>
      <th>2018</th>
      <td>42</td>
    </tr>
    <tr>
      <th>1005</th>
      <th>December</th>
      <th>2018</th>
      <td>59</td>
    </tr>
    <tr>
      <th>1006</th>
      <th>December</th>
      <th>2018</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribusi Statistik Jumlah panggilan perbulan

calls_per_month.describe()
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
      <th>calls made</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2258.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60.998671</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31.770869</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>205.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menghitung menit yang digunakan per bulan

mins_per_month = calls.groupby(['user', 'month', 'year']).agg({'duration': 'sum'}).rename(columns={'duration': 'minutes spent'})
mins_per_month.head(10)
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
      <th></th>
      <th></th>
      <th>minutes spent</th>
    </tr>
    <tr>
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <th>December</th>
      <th>2018</th>
      <td>124</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1001</th>
      <th>August</th>
      <th>2018</th>
      <td>182</td>
    </tr>
    <tr>
      <th>December</th>
      <th>2018</th>
      <td>412</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>426</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>393</td>
    </tr>
    <tr>
      <th>September</th>
      <th>2018</th>
      <td>315</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1002</th>
      <th>December</th>
      <th>2018</th>
      <td>384</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>386</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>59</td>
    </tr>
    <tr>
      <th>1003</th>
      <th>December</th>
      <th>2018</th>
      <td>1104</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribusi Statistik jumlah menit yang digunakan untuk panggilan per bulan

mins_per_month.describe()
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
      <th>minutes spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2258.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>435.937555</td>
    </tr>
    <tr>
      <th>std</th>
      <td>231.972343</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>275.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>429.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>574.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1510.000000</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan: 

1. Dari jumlah panggilan yang dilakukan perbulan, kita dapat melihat jumlah panggilan yang dilakukan 2258. user dengan panggilan terlama dengan 205 user yang menghabiskan menghabiskan 1510 menit panggilan, dan panggilan tersingkat hanya 1 panggilan dimna lama nya 0 menit dimana kemungkinan panggilan tersebut adalah panggilan tidak terjawab. 
______________________
2.  Karena rata-rata menit yang dihabiskan lebih besar dari median, kita berharap distribusinya miring ke kanan. Ini berarti data bisa mengandung outlier.

#### Jumlah SMS yang dikirim per bulan


```python
# Menghitung jumlah SMS yang dikirim per_bulan 

messages_per_month = messages.groupby(['user', 'month', 'year']).agg({'id': 'count' }).rename(columns={'id': 'messages sent'})
messages_per_month.head(10)
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
      <th></th>
      <th></th>
      <th>messages sent</th>
    </tr>
    <tr>
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <th>December</th>
      <th>2018</th>
      <td>11</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1001</th>
      <th>August</th>
      <th>2018</th>
      <td>30</td>
    </tr>
    <tr>
      <th>December</th>
      <th>2018</th>
      <td>44</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>36</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>53</td>
    </tr>
    <tr>
      <th>September</th>
      <th>2018</th>
      <td>44</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">1002</th>
      <th>December</th>
      <th>2018</th>
      <td>41</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>32</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>15</td>
    </tr>
    <tr>
      <th>1003</th>
      <th>December</th>
      <th>2018</th>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribusi statistik jumlah SMS yang dikirim per_bulan

messages_per_month.describe()
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
      <th>messages sent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1806.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>42.110188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>33.122931</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>59.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>266.000000</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

1. Rata-rata, sekitar 1806 pesan dikirim pengguna per bulan. Jumlah pesan terkirim paling sedikit adalah 1 sedangkan pesan terkirim paling banyak adalah 266. dengan rata-rata pesan terkirim 42.

#### Volume data per bulan


```python
# Penggunaan besaran data internet per_bulan

internet_per_month = (internet.groupby(['user', 'month', 'year']).agg({'data_used': 'sum'})/1024).apply(np.ceil)*1024
internet_per_month
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
      <th></th>
      <th></th>
      <th>data_used</th>
    </tr>
    <tr>
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <th>December</th>
      <th>2018</th>
      <td>2048.0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">1001</th>
      <th>August</th>
      <th>2018</th>
      <td>7168.0</td>
    </tr>
    <tr>
      <th>December</th>
      <th>2018</th>
      <td>19456.0</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>19456.0</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>22528.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1498</th>
      <th>September</th>
      <th>2018</th>
      <td>23552.0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">1499</th>
      <th>December</th>
      <th>2018</th>
      <td>22528.0</td>
    </tr>
    <tr>
      <th>November</th>
      <th>2018</th>
      <td>17408.0</td>
    </tr>
    <tr>
      <th>October</th>
      <th>2018</th>
      <td>20480.0</td>
    </tr>
    <tr>
      <th>September</th>
      <th>2018</th>
      <td>13312.0</td>
    </tr>
  </tbody>
</table>
<p>2277 rows × 1 columns</p>
</div>




```python
# Distribusi statistik penggunaan data pengguna internet

internet_per_month.describe()
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
      <th>data_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2277.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17376.070268</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7871.547017</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1024.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13312.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17408.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>21504.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>71680.000000</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

1. Dari Distribusi statistik penggunaan data internet per_bulan total pengguna data internet adalah 2277, dengan rata-rata pengguanaan data internet sekitar 17.397MB. lalu pengguna yang menggunakan data internet terendah adalah 1024MB. serta penggunaan data internet paling tinggi adalah 71.680MB.

#### Pendapatan bulanan dari setiap pengguna (kurangi batas paket gratis dari jumlah total panggilan, SMS, dan data; kalikan hasilnya dengan nilai paket telepon; tambahkan biaya bulanan tergantung pada paket teleponnya)


```python
# Menggabungkan Dataset 
agg_data = pd.concat([calls_per_month, mins_per_month, messages_per_month, internet_per_month], axis=1).reset_index().fillna(0)
agg_data.columns = ['user', 'month', 'year', 'calls_made', 'call_duration', 'messages_sent', 'mb_used']
agg_data
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
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>December</td>
      <td>2018</td>
      <td>16.0</td>
      <td>124.0</td>
      <td>11.0</td>
      <td>2048.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>August</td>
      <td>2018</td>
      <td>27.0</td>
      <td>182.0</td>
      <td>30.0</td>
      <td>7168.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>December</td>
      <td>2018</td>
      <td>56.0</td>
      <td>412.0</td>
      <td>44.0</td>
      <td>19456.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>November</td>
      <td>2018</td>
      <td>64.0</td>
      <td>426.0</td>
      <td>36.0</td>
      <td>19456.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>October</td>
      <td>2018</td>
      <td>65.0</td>
      <td>393.0</td>
      <td>53.0</td>
      <td>22528.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>1498</td>
      <td>September</td>
      <td>2018</td>
      <td>45.0</td>
      <td>363.0</td>
      <td>0.0</td>
      <td>23552.0</td>
    </tr>
    <tr>
      <th>2289</th>
      <td>1499</td>
      <td>December</td>
      <td>2018</td>
      <td>65.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>22528.0</td>
    </tr>
    <tr>
      <th>2290</th>
      <td>1499</td>
      <td>November</td>
      <td>2018</td>
      <td>45.0</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>17408.0</td>
    </tr>
    <tr>
      <th>2291</th>
      <td>1499</td>
      <td>October</td>
      <td>2018</td>
      <td>53.0</td>
      <td>385.0</td>
      <td>0.0</td>
      <td>20480.0</td>
    </tr>
    <tr>
      <th>2292</th>
      <td>1499</td>
      <td>September</td>
      <td>2018</td>
      <td>41.0</td>
      <td>346.0</td>
      <td>0.0</td>
      <td>13312.0</td>
    </tr>
  </tbody>
</table>
<p>2293 rows × 7 columns</p>
</div>




```python
# Menggabungkan dataset users
agg_data = agg_data.merge(users, on='user')
agg_data
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
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>subscription_date</th>
      <th>plan</th>
      <th>churn_date</th>
      <th>customer_churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>December</td>
      <td>2018</td>
      <td>16.0</td>
      <td>124.0</td>
      <td>11.0</td>
      <td>2048.0</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>Atlanta-Sandy Springs-Roswell, GA MSA</td>
      <td>2018-12-24</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>August</td>
      <td>2018</td>
      <td>27.0</td>
      <td>182.0</td>
      <td>30.0</td>
      <td>7168.0</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>December</td>
      <td>2018</td>
      <td>56.0</td>
      <td>412.0</td>
      <td>44.0</td>
      <td>19456.0</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>November</td>
      <td>2018</td>
      <td>64.0</td>
      <td>426.0</td>
      <td>36.0</td>
      <td>19456.0</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>October</td>
      <td>2018</td>
      <td>65.0</td>
      <td>393.0</td>
      <td>53.0</td>
      <td>22528.0</td>
      <td>Mickey</td>
      <td>Wilkerson</td>
      <td>28</td>
      <td>Seattle-Tacoma-Bellevue, WA MSA</td>
      <td>2018-08-13</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>1498</td>
      <td>September</td>
      <td>2018</td>
      <td>45.0</td>
      <td>363.0</td>
      <td>0.0</td>
      <td>23552.0</td>
      <td>Scot</td>
      <td>Williamson</td>
      <td>51</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA MSA</td>
      <td>2018-02-04</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2289</th>
      <td>1499</td>
      <td>December</td>
      <td>2018</td>
      <td>65.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>22528.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-05-06</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2290</th>
      <td>1499</td>
      <td>November</td>
      <td>2018</td>
      <td>45.0</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>17408.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-05-06</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2291</th>
      <td>1499</td>
      <td>October</td>
      <td>2018</td>
      <td>53.0</td>
      <td>385.0</td>
      <td>0.0</td>
      <td>20480.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-05-06</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2292</th>
      <td>1499</td>
      <td>September</td>
      <td>2018</td>
      <td>41.0</td>
      <td>346.0</td>
      <td>0.0</td>
      <td>13312.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-05-06</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>2293 rows × 15 columns</p>
</div>




```python
# Menggabungkan dataset plans

agg_data = agg_data.merge(plans, left_on='plan', right_on='plan')
agg_data
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
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>...</th>
      <th>plan</th>
      <th>churn_date</th>
      <th>customer_churn</th>
      <th>messages_included</th>
      <th>data_volume_per_month</th>
      <th>minutes_included</th>
      <th>monthly_fee</th>
      <th>price_per_gb</th>
      <th>price_per_message</th>
      <th>price_per_minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>December</td>
      <td>2018</td>
      <td>16.0</td>
      <td>124.0</td>
      <td>11.0</td>
      <td>2048.0</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>...</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1006</td>
      <td>December</td>
      <td>2018</td>
      <td>9.0</td>
      <td>59.0</td>
      <td>139.0</td>
      <td>32768.0</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>...</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
      <td>Yes</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1006</td>
      <td>November</td>
      <td>2018</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>3072.0</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>...</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
      <td>Yes</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1008</td>
      <td>December</td>
      <td>2018</td>
      <td>85.0</td>
      <td>634.0</td>
      <td>26.0</td>
      <td>15360.0</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>...</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1008</td>
      <td>November</td>
      <td>2018</td>
      <td>63.0</td>
      <td>446.0</td>
      <td>37.0</td>
      <td>24576.0</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>...</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>1498</td>
      <td>September</td>
      <td>2018</td>
      <td>45.0</td>
      <td>363.0</td>
      <td>0.0</td>
      <td>23552.0</td>
      <td>Scot</td>
      <td>Williamson</td>
      <td>51</td>
      <td>...</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2289</th>
      <td>1499</td>
      <td>December</td>
      <td>2018</td>
      <td>65.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>22528.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>...</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2290</th>
      <td>1499</td>
      <td>November</td>
      <td>2018</td>
      <td>45.0</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>17408.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>...</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2291</th>
      <td>1499</td>
      <td>October</td>
      <td>2018</td>
      <td>53.0</td>
      <td>385.0</td>
      <td>0.0</td>
      <td>20480.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>...</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>2292</th>
      <td>1499</td>
      <td>September</td>
      <td>2018</td>
      <td>41.0</td>
      <td>346.0</td>
      <td>0.0</td>
      <td>13312.0</td>
      <td>Shena</td>
      <td>Dickson</td>
      <td>37</td>
      <td>...</td>
      <td>surf</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>50</td>
      <td>15360</td>
      <td>500</td>
      <td>20</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
<p>2293 rows × 22 columns</p>
</div>




```python
# Menentukan Pendapatan yang didapat dari setiap pengguna dari jumlah panggilan 

agg_data['call_cost'] = agg_data.apply(lambda x: max(0, x['call_duration'] - x['minutes_included']) * x['price_per_minute'], axis = 1)
```


```python
# Menentukan Pendapatan yang didapat dari setiap pengguna dari SMS

agg_data['message_cost'] = agg_data.apply(lambda x: max(0, x['messages_sent'] - x['messages_included']) * x['price_per_message'], axis = 1)
```


```python
# Menentukan Pendapatan yang didapat dari setiap pengguna dari data Internet

agg_data['gb_cost'] = agg_data.apply(lambda x: np.ceil(max(0, x['mb_used'] - x['data_volume_per_month'])/1024)* x['price_per_gb'], axis = 1)
```


```python
pd.set_option('display.max_columns', 100)
```


```python
# Meghitung pendapatan dari jumlah biaya panggilan, jumlah biaya pesan, dan jumlah biaya data internet

agg_data['revenue'] = agg_data['call_cost'] + agg_data['message_cost'] + agg_data['gb_cost'] + agg_data['monthly_fee']
agg_data.head()
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
      <th>user</th>
      <th>month</th>
      <th>year</th>
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>age</th>
      <th>city</th>
      <th>subscription_date</th>
      <th>plan</th>
      <th>churn_date</th>
      <th>customer_churn</th>
      <th>messages_included</th>
      <th>data_volume_per_month</th>
      <th>minutes_included</th>
      <th>monthly_fee</th>
      <th>price_per_gb</th>
      <th>price_per_message</th>
      <th>price_per_minute</th>
      <th>call_cost</th>
      <th>message_cost</th>
      <th>gb_cost</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>December</td>
      <td>2018</td>
      <td>16.0</td>
      <td>124.0</td>
      <td>11.0</td>
      <td>2048.0</td>
      <td>Anamaria</td>
      <td>Bauer</td>
      <td>45</td>
      <td>Atlanta-Sandy Springs-Roswell, GA MSA</td>
      <td>2018-12-24</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1006</td>
      <td>December</td>
      <td>2018</td>
      <td>9.0</td>
      <td>59.0</td>
      <td>139.0</td>
      <td>32768.0</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>San Francisco-Oakland-Berkeley, CA MSA</td>
      <td>2018-11-27</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
      <td>Yes</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1006</td>
      <td>November</td>
      <td>2018</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>3072.0</td>
      <td>Jesusa</td>
      <td>Bradford</td>
      <td>73</td>
      <td>San Francisco-Oakland-Berkeley, CA MSA</td>
      <td>2018-11-27</td>
      <td>ultimate</td>
      <td>2018-12-18</td>
      <td>Yes</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1008</td>
      <td>December</td>
      <td>2018</td>
      <td>85.0</td>
      <td>634.0</td>
      <td>26.0</td>
      <td>15360.0</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-08-03</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1008</td>
      <td>November</td>
      <td>2018</td>
      <td>63.0</td>
      <td>446.0</td>
      <td>37.0</td>
      <td>24576.0</td>
      <td>Emely</td>
      <td>Hoffman</td>
      <td>53</td>
      <td>Orlando-Kissimmee-Sanford, FL MSA</td>
      <td>2018-08-03</td>
      <td>ultimate</td>
      <td>2018-01-01</td>
      <td>No</td>
      <td>1000</td>
      <td>30720</td>
      <td>3000</td>
      <td>70</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribusi statistik pada jumlah panggilan yang dilakukan dan menit yang digunakan per bulan,
# Jumlah SMS yang dikirim per bulan,
# Volume data per bulan,
# Pendapatan bulanan dari setiap pengguna,

agg_data.describe()
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
      <th>user</th>
      <th>year</th>
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>age</th>
      <th>messages_included</th>
      <th>data_volume_per_month</th>
      <th>minutes_included</th>
      <th>monthly_fee</th>
      <th>price_per_gb</th>
      <th>price_per_message</th>
      <th>price_per_minute</th>
      <th>call_cost</th>
      <th>message_cost</th>
      <th>gb_cost</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2293.000000</td>
      <td>2293.0</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
      <td>2293.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1246.075883</td>
      <td>2018.0</td>
      <td>60.067597</td>
      <td>429.283471</td>
      <td>33.166594</td>
      <td>17254.824248</td>
      <td>45.428260</td>
      <td>348.299171</td>
      <td>20183.026603</td>
      <td>1284.997819</td>
      <td>35.699956</td>
      <td>9.058003</td>
      <td>0.023720</td>
      <td>0.023720</td>
      <td>1.264828</td>
      <td>0.144322</td>
      <td>27.144352</td>
      <td>64.253458</td>
    </tr>
    <tr>
      <th>std</th>
      <td>143.051927</td>
      <td>0.0</td>
      <td>32.402563</td>
      <td>236.320077</td>
      <td>34.070085</td>
      <td>7976.321502</td>
      <td>16.764349</td>
      <td>441.006389</td>
      <td>7130.376976</td>
      <td>1160.543128</td>
      <td>23.210863</td>
      <td>1.392652</td>
      <td>0.009284</td>
      <td>0.009284</td>
      <td>3.233992</td>
      <td>0.493515</td>
      <td>48.444483</td>
      <td>46.514549</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>2018.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>50.000000</td>
      <td>15360.000000</td>
      <td>500.000000</td>
      <td>20.000000</td>
      <td>7.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1122.000000</td>
      <td>2018.0</td>
      <td>38.000000</td>
      <td>265.000000</td>
      <td>3.000000</td>
      <td>12288.000000</td>
      <td>30.000000</td>
      <td>50.000000</td>
      <td>15360.000000</td>
      <td>500.000000</td>
      <td>20.000000</td>
      <td>7.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>23.480000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1245.000000</td>
      <td>2018.0</td>
      <td>60.000000</td>
      <td>425.000000</td>
      <td>26.000000</td>
      <td>17408.000000</td>
      <td>46.000000</td>
      <td>50.000000</td>
      <td>15360.000000</td>
      <td>500.000000</td>
      <td>20.000000</td>
      <td>10.000000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1368.000000</td>
      <td>2018.0</td>
      <td>79.000000</td>
      <td>572.000000</td>
      <td>51.000000</td>
      <td>21504.000000</td>
      <td>61.000000</td>
      <td>1000.000000</td>
      <td>30720.000000</td>
      <td>3000.000000</td>
      <td>70.000000</td>
      <td>10.000000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1499.000000</td>
      <td>2018.0</td>
      <td>205.000000</td>
      <td>1510.000000</td>
      <td>266.000000</td>
      <td>71680.000000</td>
      <td>75.000000</td>
      <td>1000.000000</td>
      <td>30720.000000</td>
      <td>3000.000000</td>
      <td>70.000000</td>
      <td>10.000000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>30.300000</td>
      <td>6.480000</td>
      <td>550.000000</td>
      <td>590.370000</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

1. Pada kesimpulan ini, kita telah menyiapkan dataset untuk dianalis dengan dengan menggabungkan Jumlah panggilan yang dilakukan dan menit yang digunakan per bulan, Jumlah SMS yang dikirim per bulan, Jumlah penggunaan Volume data internet per bulan.
_____________________________

2. lalu menghitung Pendapatan keuntungan dari jumlah biaya panggilan, jumlah biaya pesan, dan jumlah biaya data internet .

## Data Analyzing

#### Analisis data exploratory 


```python
# Mengelompokkan value dari data numerik dan categorical
numerical_list = []
categorical_list = []
plot_data = agg_data[['calls_made', 'call_duration', 'messages_sent', 'mb_used', 'plan', 'call_cost', 'gb_cost', 'message_cost', 'revenue']]

for column in plot_data:
    if is_numeric_dtype(plot_data[column]):
        numerical_list.append(column)
    elif is_string_dtype(plot_data[column]):
        categorical_list.append(column)
        
print('list dari tipe data numerical :', numerical_list)
print('list dari tipe data categorical : ',categorical_list)
```

    list dari tipe data numerical : ['calls_made', 'call_duration', 'messages_sent', 'mb_used', 'call_cost', 'gb_cost', 'message_cost', 'revenue']
    list dari tipe data categorical :  ['plan']


##### Histogram call_made


```python
# Tabel Histogram dari jumlah panggilan dari pengguna
agg_data['calls_made'].hist()

# menambahkan judul dan nama sumbu 
plt.xlabel('call_made')
plt.ylabel('Frequency')
plt.title("Histogram dari jumlah panggilan");
```


    
![png](output_112_0.png)
    



```python
# Distribusi statistik pada column jumlah panggilan
agg_data['calls_made'].describe()
```




    count    2293.000000
    mean       60.067597
    std        32.402563
    min         0.000000
    25%        38.000000
    50%        60.000000
    75%        79.000000
    max       205.000000
    Name: calls_made, dtype: float64




```python
# Menghitung rata-rata, varians, dan standar deviasi

agg_data.groupby('plan')['calls_made'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>59.811825</td>
      <td>1025.15159</td>
      <td>32.017989</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>60.626389</td>
      <td>1105.09666</td>
      <td>33.242994</td>
    </tr>
  </tbody>
</table>
</div>



<div class="alert alert-success">
<b>Reviewer's comment</b> <a class="tocSkip"></a>

sudah benar yah!

</div>

##### Histogram call_duration


```python
# Tabel Histogram dari durasi panggilan 
agg_data['call_duration'].hist()

# menambahkan judul dan nama sumbu 
plt.xlabel('call_duration')
plt.ylabel('Frequency')
plt.title("Histogram dari durasi panggilan");
```


    
![png](output_117_0.png)
    



```python
# Distribusi statistik pada column call duration
agg_data['call_duration'].describe()
```




    count    2293.000000
    mean      429.283471
    std       236.320077
    min         0.000000
    25%       265.000000
    50%       425.000000
    75%       572.000000
    max      1510.000000
    Name: call_duration, dtype: float64




```python
# Menghitung rata-rata, varians, dan standar deviasi

agg_data.groupby('plan')['call_duration'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>428.749523</td>
      <td>54968.279461</td>
      <td>234.453150</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>430.450000</td>
      <td>57844.464812</td>
      <td>240.508762</td>
    </tr>
  </tbody>
</table>
</div>



<div class="alert alert-success">
<b>Reviewer's comment</b> <a class="tocSkip"></a>

sudah benar yah!

</div>

##### Histogram messages_sent


```python
# Tabel Histogram dari jumlah pesan terkirim
agg_data['messages_sent'].hist()

# menambahkan judul dan nama sumbu 
plt.xlabel('call_duration')
plt.ylabel('Frequency')
plt.title("Histogram dari jumlah pesan");
```


    
![png](output_122_0.png)
    



```python
# Distribusi statistik pada column messages_sent
agg_data['messages_sent'].describe()
```




    count    2293.000000
    mean       33.166594
    std        34.070085
    min         0.000000
    25%         3.000000
    50%        26.000000
    75%        51.000000
    max       266.000000
    Name: messages_sent, dtype: float64




```python
# Menghitung rata-rata, varians, dan standar deviasi

agg_data.groupby('plan')['messages_sent'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>31.159568</td>
      <td>1126.724522</td>
      <td>33.566717</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>37.551389</td>
      <td>1208.756744</td>
      <td>34.767179</td>
    </tr>
  </tbody>
</table>
</div>



<div class="alert alert-success">
<b>Reviewer's comment</b> <a class="tocSkip"></a>

sudah benar yah!

</div>

##### Histogram mb_used


```python
# Tabel Histogram dari jumlah penggunaan volume data internet 
agg_data['mb_used'].hist()

# menambahkan judul dan nama sumbu 
plt.xlabel('mb_used')
plt.ylabel('Frequency')
plt.title("Histogram dari jumlah volume penggunaan data internet");
```


    
![png](output_127_0.png)
    



```python
# Distribusi statistik pada column messages_sent
agg_data['mb_used'].describe()
```




    count     2293.000000
    mean     17254.824248
    std       7976.321502
    min          0.000000
    25%      12288.000000
    50%      17408.000000
    75%      21504.000000
    max      71680.000000
    Name: mb_used, dtype: float64



<div class="alert alert-success">
<b>Reviewer's comment</b> <a class="tocSkip"></a>

sudah benar yah!

</div>

##### Bar Plot Plan


```python
# Tabel bar dari jumlah penggunaan tipe paket  
agg_data['plan'].value_counts().plot(kind = 'bar')

# menambahkan judul dan nama sumbu 
plt.xlabel('Plan')
plt.ylabel('Frequency')
plt.title("Bar Plot dari jumlah penggunaan tipe paket");
plt.show()
```


    
![png](output_131_0.png)
    



```python
# Menghitung rata-rata, varians, dan standar deviasi
agg_data['plan'].describe()
```




    count     2293
    unique       2
    top       surf
    freq      1573
    Name: plan, dtype: object



<div class="alert alert-success">
<b>Reviewer's comment</b> <a class="tocSkip"></a>

sudah benar yah!

</div>

##### Histogram Revenue


```python
# Tabel Histogram dari jumlah penggunaan volume data internet 
agg_data['revenue'].hist()

# menambahkan judul dan nama sumbu 
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.title("Histogram dari pendapatan");
```


    
![png](output_135_0.png)
    



```python
# Distribusi statistik pada column messages_sent

agg_data['revenue'].describe()
```




    count    2293.000000
    mean       64.253458
    std        46.514549
    min        20.000000
    25%        23.480000
    50%        70.000000
    75%        70.000000
    max       590.370000
    Name: revenue, dtype: float64




```python
# Menghitung rata-rata, varians, dan standar deviasi

agg_data.groupby('plan')['revenue'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>60.572905</td>
      <td>3052.714450</td>
      <td>55.251375</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>72.294444</td>
      <td>128.302612</td>
      <td>11.327074</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan  :

1.  Pada tahap analisis ini, kita menggunakan histogram dan bar plot untuk membuat tabel grafik dari variabel numerik dan kategori. pada distribusi frekuensi dari tabel grafik diatas menampilkan frekuensi data dari keseluruhan data. semua grafik diatas menunjukkan bahwa distribusinya agak miring kekanan. 
_________________________________

2. Tetapi dalam histogram messages_sent, dimana pada pada plan ultimate diketahui terdapat 166 pesan maksimal yang terkirim dan 0 pesan minimum yang terkirim dengan standar deviasiny 34 dimana meannya jauh dari median dimana jumlah mean 31 dan mediannya 24. kemungikinan data mememiliki outlier.

________________________________

3. Pada grafik plot batang dari nama paket menunjukkan bahwa pengguna paket dengan frekuensi penggunaan terbanyak pada surf 1573 dan penggunaan paket ultimate 720. dapat disimpulkan lebih banyak pengguna menggunakan paket surf dari pada ultimate. kita perlu menganalisis lebih dalam dari data ini untuk menentukan paket mana yang lebih banyak keuntungan pendapatan tiap bulan. 

_________________________________

4. Pada Tahap selanjutnya kita akan melakuakn identifikasi dan memfiter outlier dalam data.

#### Mempelajari dan Menangani Outlier

Dengan menggunakan metode IQR  ukuran variabilitas yang didasarkan pada pembagian kumpulan data menjadi kuartil. Kuartil membagi kumpulan data terurut menjadi empat bagian yang sama besar. Nilai yang memisahkan bagian-bagian ini disebut kuartil pertama, kedua (median), dan ketiga yang masing-masing dilambangkan dengan Q1, Q2, dan Q3

dengan menggunakan metode ini kita dapat mendeteksi nilai atas dan bawah dari outlier serta memfilter data yang teridentifikasi berisi outlier. 

![image.png](attachment:image.png)


```python
# fungsi menentukan batas bawah outlier

def lower_whisker(data, col):
    q1 = agg_data[col].quantile(0.25)
    q3 = agg_data[col].quantile(0.75)
    
    iqr = q3 - q1
    
    return q1 - 1.5 * iqr
```


```python
# fungsi menentukan batas atas outlier

def upper_whisker(data, col):
    q1 = agg_data[col].quantile(0.25)
    q3 = agg_data[col].quantile(0.75)
    
    iqr = q3 - q1
    
    return q3 + 1.5 * iqr
```


```python
# batas bawah calls_made

lower_call_made = lower_whisker(agg_data, 'calls_made')
lower_call_made
```




    -23.5




```python
# batas atas calls_made

upper_call_made = upper_whisker(agg_data, 'calls_made')
upper_call_made
```




    140.5




```python
# batas bawah calls_duration

lower_call_duration = lower_whisker(agg_data, 'call_duration')
lower_call_duration
```




    -195.5




```python
# batas atas calls_duration

upper_call_duration = upper_whisker(agg_data, 'call_duration')
upper_call_duration
```




    1032.5




```python
# batas bawah messages_sent

lower_messages_sent = lower_whisker(agg_data, 'messages_sent')
lower_messages_sent

```




    -69.0




```python
# batas atas messages_sent

upper_messages_sent = upper_whisker(agg_data, 'messages_sent')
upper_messages_sent
```




    123.0




```python
# batas bawah mb_used

lower_mb_used = lower_whisker(agg_data, 'mb_used')
lower_mb_used

```




    -1536.0




```python
# batas atas mb_used

upper_mb_used = upper_whisker(agg_data, 'mb_used')
upper_mb_used
```




    35328.0




```python
# batas bawah plan

lower_revenue = lower_whisker(agg_data, 'revenue')
lower_revenue
```




    -46.3




```python
# batas atas plan

upper_revenue = upper_whisker(agg_data, 'revenue')
upper_revenue
```




    139.78




```python
# Memuat dataset baru 

telecom = agg_data[['calls_made', 'call_duration', 'messages_sent', 'mb_used', 'revenue']]
```


```python
# memfilter dataset baru sesuai dengan memfilter data outliernya

telecom_filtered = telecom.query('(calls_made > @lower_call_made and calls_made < @upper_call_made) and (call_duration > @lower_call_duration and call_duration < @upper_call_duration) and (messages_sent > @lower_messages_sent and messages_sent < @upper_messages_sent) and (mb_used > @lower_mb_used and mb_used < @upper_mb_used) and (revenue > @lower_revenue and revenue < @upper_revenue)')
```

#### Grafik Histogram dari dataset yang telah di filter outliernya


```python
# Membuat grafik histogram dari dataset yang telah di filter outliernya

telecom_filtered[['calls_made', 'call_duration', 'messages_sent', 'mb_used', 'revenue']].hist(bins=30, figsize=(15, 10))
plt.suptitle('histogram dari dataset yang telah di filter outliernya', y=0.95);
```


    
![png](output_156_0.png)
    


####  Boxplot 


```python
# Boxplot dari jumlah panggilan dari pengguna
plt.figure(figsize=(5,5))
sns.boxplot(data=telecom_filtered['calls_made'])

plt.title('Boxplot dari jumlah panggilan dari pengguna ')
plt.suptitle("")
plt.show()
```


    
![png](output_158_0.png)
    



```python
# Boxplot dari durasi panggilan 
plt.figure(figsize=(5,5))
sns.boxplot(data=telecom_filtered['call_duration'] )

plt.title('Boxplot dari durasi panggilan ')
plt.suptitle("")
plt.show()
```


    
![png](output_159_0.png)
    



```python
# Boxplot dari jumlah pesan terkirim 
plt.figure(figsize=(5,5))
sns.boxplot(data=telecom_filtered['messages_sent'] )

plt.title('Boxplot dari jumlah pesan terkirim ')
plt.suptitle("")
plt.show()
```


    
![png](output_160_0.png)
    



```python
# Boxplot dari jumlah penggunaan volume data internet 
plt.figure(figsize=(5,5))
sns.boxplot(data=telecom_filtered['mb_used'] )

plt.title('Boxplot dari jumlah penggunaan volume data internet  ')
plt.suptitle("")
plt.show()
```


    
![png](output_161_0.png)
    



```python
# Boxplot dari jumlah penggunaan volume data internet 
plt.figure(figsize=(5,5))
sns.boxplot(data=telecom_filtered['revenue'],  )

plt.title('Boxplot dari jumlah penggunaan volume data internet  ')
plt.suptitle("")
plt.show()
```


    
![png](output_162_0.png)
    



```python
telecom_filtered
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
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.0</td>
      <td>124.0</td>
      <td>11.0</td>
      <td>2048.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>3072.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85.0</td>
      <td>634.0</td>
      <td>26.0</td>
      <td>15360.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63.0</td>
      <td>446.0</td>
      <td>37.0</td>
      <td>24576.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>71.0</td>
      <td>476.0</td>
      <td>21.0</td>
      <td>17408.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2288</th>
      <td>45.0</td>
      <td>363.0</td>
      <td>0.0</td>
      <td>23552.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2289</th>
      <td>65.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>22528.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>2290</th>
      <td>45.0</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>17408.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>2291</th>
      <td>53.0</td>
      <td>385.0</td>
      <td>0.0</td>
      <td>20480.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2292</th>
      <td>41.0</td>
      <td>346.0</td>
      <td>0.0</td>
      <td>13312.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
<p>2054 rows × 5 columns</p>
</div>




```python
# # distribusi statistik pada dataset yang telah di filter nilai outliernya

telecom_filtered.describe()
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
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2054.000000</td>
      <td>2054.000000</td>
      <td>2054.000000</td>
      <td>2054.000000</td>
      <td>2054.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.994158</td>
      <td>407.346641</td>
      <td>28.805258</td>
      <td>15969.215190</td>
      <td>54.985604</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.766879</td>
      <td>209.955720</td>
      <td>27.826896</td>
      <td>6437.306912</td>
      <td>28.448963</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>36.000000</td>
      <td>257.250000</td>
      <td>3.000000</td>
      <td>12288.000000</td>
      <td>21.230000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>58.000000</td>
      <td>413.000000</td>
      <td>24.000000</td>
      <td>16384.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.000000</td>
      <td>550.000000</td>
      <td>46.000000</td>
      <td>20480.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.000000</td>
      <td>1029.000000</td>
      <td>121.000000</td>
      <td>34816.000000</td>
      <td>137.740000</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan : 

1. Dengan menggunakan metode statistik IQR untuk menentukan distribusi nilai baku untuk menentukan rentang data normal dimana nilai nya dibagi menjadi 3 kuartil yakni q1 untuk nilai bawah , q2 untuk median, dan q3 untuk batas atas dari nilai outlier nya. sehingga sebaran data menjadi normal


#### Mempelajari Perilaku konsumen  :

1. menghitung jumlah panggilan yang dilakukan dan menit yang digunakan perbulan, Jumlah SMS yang dikirim per bulan Penggunaan volume data internet per bulan yang dibutuhkan costumer pada paket per bulan.  

##### Menghitung jumlah panggilan yang dilakukan dan menit yang digunakan perbulan


```python
# Durasi pengguna pada paket yang digunakan perbulan 

agg_data.groupby('plan')['call_duration'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>428.749523</td>
      <td>54968.279461</td>
      <td>234.453150</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>430.450000</td>
      <td>57844.464812</td>
      <td>240.508762</td>
    </tr>
  </tbody>
</table>
</div>



Apa perbedaan rata-rata durasi panggilan bulanan untuk pelanggan di kedua paket?


```python
# Perbedaan rata-rata penggunaan durasi panggilan perbulan di kedua paket

agg_data.groupby('plan')['call_duration'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>plan</th>
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
      <th>surf</th>
      <td>1573.0</td>
      <td>428.749523</td>
      <td>234.453150</td>
      <td>0.0</td>
      <td>272.0</td>
      <td>425.0</td>
      <td>576.00</td>
      <td>1510.0</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>720.0</td>
      <td>430.450000</td>
      <td>240.508762</td>
      <td>0.0</td>
      <td>260.0</td>
      <td>424.0</td>
      <td>565.25</td>
      <td>1369.0</td>
    </tr>
  </tbody>
</table>
</div>



Pada distribusi diatas, kita dapat mengetahui dari rata-rata paket ultimate lebih banyak penggunaan durasi panggilan  perbulan daripada pengguna paket surf, mari kita lihat apakah hasil diatas akan sama ketika kita menghilangkan data ouliernya. 


```python
# Filter data untuk paket ultimate

ultimate_data = agg_data.query("plan == 'ultimate'")
ultimate_plan = ultimate_data[['calls_made', 'call_duration', 'messages_sent', 'mb_used', 'revenue']]

ultimate_plan_filtered = ultimate_plan.query('(calls_made > @lower_call_made and calls_made < @upper_call_made) and (call_duration > @lower_call_duration and call_duration < @upper_call_duration) and (messages_sent > @lower_messages_sent and messages_sent < @upper_messages_sent) and (mb_used > @lower_mb_used and mb_used < @upper_mb_used) and (revenue > @lower_revenue and revenue < @upper_revenue)')
ultimate_plan_filtered.describe()
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
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>660.000000</td>
      <td>660.000000</td>
      <td>660.000000</td>
      <td>660.000000</td>
      <td>660.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>58.078788</td>
      <td>411.772727</td>
      <td>32.674242</td>
      <td>16653.963636</td>
      <td>70.360606</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.609078</td>
      <td>207.147107</td>
      <td>29.298499</td>
      <td>6754.406703</td>
      <td>2.618997</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>36.000000</td>
      <td>260.000000</td>
      <td>5.000000</td>
      <td>13312.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.000000</td>
      <td>418.500000</td>
      <td>28.000000</td>
      <td>16384.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.000000</td>
      <td>541.250000</td>
      <td>54.000000</td>
      <td>20480.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.000000</td>
      <td>987.000000</td>
      <td>121.000000</td>
      <td>34816.000000</td>
      <td>98.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# variance dari paket ultimate 
for column in ultimate_plan_filtered:
    ultimate_plan_filtered[column].var()
    print('Variance dari ' + column + ' adalah {: >5.2f}'.format(ultimate_plan_filtered[column].var()))
```

    Variance dari calls_made adalah 818.48
    Variance dari call_duration adalah 42909.92
    Variance dari messages_sent adalah 858.40
    Variance dari mb_used adalah 45622009.90
    Variance dari revenue adalah  6.86


Dari Deskripsi diatas, kita menentukan bahwa pengguna paket ultimate rata-rata melakukan 58 panggilan, dan durasi nya 411 menit perbulan, dan mengirim pesan 32 perbulan, menggunakan data 16645 MB data perbulan, dengan varians paket ultimate sebesar 819 calls_made dan 859 messages_sent.


```python
# Filter data untuk paket surf 

surf_data = agg_data.query("plan == 'surf'")
surf_plan = surf_data[['calls_made', 'call_duration', 'messages_sent', 'mb_used', 'revenue']]

surf_plan_filtered = surf_plan.query('(calls_made > @lower_call_made and calls_made < @upper_call_made) and (call_duration > @lower_call_duration and call_duration < @upper_call_duration) and (messages_sent > @lower_messages_sent and messages_sent < @upper_messages_sent) and (mb_used > @lower_mb_used and mb_used < @upper_mb_used) and (revenue > @lower_revenue and revenue < @upper_revenue)')
surf_plan_filtered.describe()
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
      <th>calls_made</th>
      <th>call_duration</th>
      <th>messages_sent</th>
      <th>mb_used</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1394.000000</td>
      <td>1394.000000</td>
      <td>1394.000000</td>
      <td>1394.000000</td>
      <td>1394.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>56.480631</td>
      <td>405.251076</td>
      <td>26.973458</td>
      <td>15645.015782</td>
      <td>47.706191</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.837291</td>
      <td>211.313929</td>
      <td>26.919731</td>
      <td>6257.986770</td>
      <td>32.008347</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>37.000000</td>
      <td>255.250000</td>
      <td>2.000000</td>
      <td>12288.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>57.000000</td>
      <td>410.000000</td>
      <td>21.000000</td>
      <td>16384.000000</td>
      <td>33.865000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.000000</td>
      <td>558.000000</td>
      <td>42.000000</td>
      <td>20480.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>140.000000</td>
      <td>1029.000000</td>
      <td>121.000000</td>
      <td>26624.000000</td>
      <td>137.740000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# variance dari paket surf 
for column in surf_plan_filtered:
    surf_plan_filtered[column].var()
    print('Variance dari ' + column + ' adalah: {: >5.2f}'.format(surf_plan_filtered[column].var()))
```

    Variance dari calls_made adalah: 831.59
    Variance dari call_duration adalah: 44653.58
    Variance dari messages_sent adalah: 724.67
    Variance dari mb_used adalah: 39162398.42
    Variance dari revenue adalah: 1024.53


Pada deskripsi nilai distribusi diatas pada paket surf, kita dapat melihat rata-rata pengguna melakukan panggilan sebanyak 56, dengan durasi 405 menit perbulan, dan mengirim pesan sebanyak 26, dan menggunakan data 15662 MB, dengan varians paket untuk surf sebesar 831.59 call_made dan 724.67 messages_sent.


```python
# Histogram dari jumlah panggilan per bulan pada kedua paket
plt.figure(figsize=(6,5))
plt.hist(ultimate_plan_filtered['calls_made'], bins=20, alpha=0.5, label='ultimate')
plt.hist(surf_plan_filtered['calls_made'], bins=20, alpha=0.5, label='surf')

# menambahkan judul dan nama sumbu
plt.xlabel('Jumlah Panggilan', size=10)
plt.ylabel('Count', size=10)
plt.title('Overlay dari jumlah panggilan per bulan pada kedua paket')
plt.legend(loc='upper right');
```


    
![png](output_178_0.png)
    


Pada Overlay histogram jumlah panggilan perbulan pada paket ultimate dan surf diatas, terlihat bahwa paket surf memliki reprenstasi paling banyak untuk jumlah panggilan perbulan


```python
# Tabel Histogram dari Kepadatan frekuensi jumlah panggilan perbulan

plt.hist([ultimate_plan_filtered['calls_made'], surf_plan_filtered['calls_made']], label = ['ultimate', 'surf'], density = True)

# menambahkan judul dan nama sumbu
plt.ylabel('Frekuensi Kepadatan')
plt.xlabel('Jumlah Panggilan')
plt.title('frekuensi kepadatan dari jumlah panggilan perbulan')
plt.legend()
plt.show()
```


    
![png](output_180_0.png)
    


Pada histogram kepadatan frekuensi dengan distribusi jumlah panggilan dari pengguna per bulan untuk membandingkan nilai jumlah pengguna pada paket ultimate dan surf. dengan melihat plot histogram diatas paket surf memiliki jumlah panggilan perbulan lebih banyak dibandingkan paket ultimate, dengan plot kepadatan frekunse kita dapat melihat jumlah panggilan yang dilakukan pengguna di seluruh paket telepon. 


```python
# Histogram dari jumlah panggilan per bulan pada kedua paket
plt.figure(figsize=(6,5))
plt.hist(ultimate_plan_filtered['call_duration'], bins=20, alpha=0.5, label='ultimate')
plt.hist(surf_plan_filtered['call_duration'], bins=20, alpha=0.5, label='surf')

# menambahkan judul dan nama sumbu
plt.xlabel('Durasi Panggilan', size=10)
plt.ylabel('Count', size=10)
plt.title('Overlay dari durasi panggilan per bulan pada kedua paket')
plt.legend(loc='upper right');
```


    
![png](output_182_0.png)
    



```python
# Tabel Histogram dari Kepadatan frekuensi jumlah panggilan perbulan

plt.hist([ultimate_plan_filtered['call_duration'], surf_plan_filtered['call_duration']], label = ['ultimate', 'surf'], density = True)

# menambahkan judul dan nama sumbu
plt.ylabel('Frekuensi Kepadatan')
plt.xlabel('Durasi Panggilan')
plt.title('frekuensi kepadatan dari durasi panggilan perbulan')
plt.legend()
plt.show()
```


    
![png](output_183_0.png)
    


Dari overlay histogram plot diatas, kita dapat melihat bahwa paket ultimate  memiliki durasi panggilan per bulan lebih banyak daripada pengguna di paket surf meskipun rata-rata pengguna di paket surf melakukan lebih banyak panggilan daripada pengguna di paket ultimate.

##### Menghitung Jumlah SMS yang dikirim per bulan


```python
# Jumlah sms yang dikirim pengguna pada paket kedua paket yang  

agg_data.groupby('plan')['messages_sent'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>31.159568</td>
      <td>1126.724522</td>
      <td>33.566717</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>37.551389</td>
      <td>1208.756744</td>
      <td>34.767179</td>
    </tr>
  </tbody>
</table>
</div>



 Apa perbedaan rata-rata pesan yang dikirim pelanggan pada kedua paket?


```python
# Perbedaan rata-rata pesan yang dikirim pelanggan pada kedua paket

agg_data.groupby('plan')['messages_sent'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>plan</th>
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
      <th>surf</th>
      <td>1573.0</td>
      <td>31.159568</td>
      <td>33.566717</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>47.0</td>
      <td>266.0</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>720.0</td>
      <td>37.551389</td>
      <td>34.767179</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>30.0</td>
      <td>61.0</td>
      <td>166.0</td>
    </tr>
  </tbody>
</table>
</div>



Pada distribusi diatas kita dapat melitah jumlah pengguna dari kedua paket dimana pengguna surf lebih banyak dibandingkan ultimate. mari kita buktikan dengan histogram dengan menggunakan distribusi data pada kedua paket.


```python
# Histogram dari Jumlah Pesan Terkirim per bulan pada kedua paket
plt.figure(figsize=(6,5))
plt.hist(ultimate_plan_filtered['messages_sent'], bins=20, alpha=0.5, label='ultimate')
plt.hist(surf_plan_filtered['messages_sent'], bins=20, alpha=0.5, label='surf')

# menambahkan judul dan nama sumbu
plt.xlabel('Jumlah Pesan Terkirim', size=10)
plt.ylabel('Count', size=10)
plt.title('Overlay dari jumlah pesan terkirim per bulan pada kedua paket')
plt.legend(loc='upper right');
```


    
![png](output_190_0.png)
    



```python
# Tabel Histogram dari Kepadatan frekuensi jumlah pesan terkirim  perbulan

plt.hist([ultimate_plan_filtered['messages_sent'], surf_plan_filtered['messages_sent']], label = ['ultimate', 'surf'], density = True)

# menambahkan judul dan nama sumbu
plt.ylabel('Frekuensi Kepadatan')
plt.xlabel('jumlah pesan terkirim')
plt.title('frekuensi kepadatan dari jumlah pesan terkirim perbulan')
plt.legend()
plt.show()
```


    
![png](output_191_0.png)
    


Dari tampilan histogram yang menunjukkan distribusi total pesan terkirim per bulan, terlihat bahwa paket surf memiliki total pesan terkirim paling banyak per bulan. pada plot kepadatan frekuensi kita dapat melihat bahwa kedua paket memiliki grafik yang sama. Namun paket surf memiliki lebih banyak pengguna daripada paket ultimate.

##### Menghitung volume penggunaan data seluler yang dibutuhkan pengguna setiap paket per bulan.


```python
# Jumlah penggunaan volume data yang dibuthkan pengguna perbulan pada kedua paket

agg_data.groupby('plan')['mb_used'].agg([np.mean, np.var, np.std])
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
      <th>mean</th>
      <th>var</th>
      <th>std</th>
    </tr>
    <tr>
      <th>plan</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surf</th>
      <td>17049.958042</td>
      <td>6.444453e+07</td>
      <td>8027.734963</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>17702.400000</td>
      <td>6.161877e+07</td>
      <td>7849.762428</td>
    </tr>
  </tbody>
</table>
</div>



Apa Perbedaan rata-rata penggunaan volume data yang dibuthkan pengguna perbulan pada kedua paket?


```python
# Perbedaan rata-rata penggunaan volume data yang dibuthkan pengguna perbulan pada kedua paket

agg_data.groupby('plan')['mb_used'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>plan</th>
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
      <th>surf</th>
      <td>1573.0</td>
      <td>17049.958042</td>
      <td>8027.734963</td>
      <td>0.0</td>
      <td>12288.0</td>
      <td>17408.0</td>
      <td>21504.0</td>
      <td>71680.0</td>
    </tr>
    <tr>
      <th>ultimate</th>
      <td>720.0</td>
      <td>17702.400000</td>
      <td>7849.762428</td>
      <td>0.0</td>
      <td>13312.0</td>
      <td>17408.0</td>
      <td>21504.0</td>
      <td>47104.0</td>
    </tr>
  </tbody>
</table>
</div>



Dapat dilihata pada distribusi statistik pada penggunaan data pada kedua paket, paket ultimate lebih banyak penggunaan data dibanding dengan paket surf. mari kita lihat dengan tabel histogram untuk melihat distribusinya.


```python
# Histogram dari Jumlah penggunaan volume data yang dibuthkan pengguna perbulan pada kedua paket

plt.figure(figsize=(6,5))
plt.hist(ultimate_plan_filtered['mb_used'], bins=20, alpha=0.5, label='ultimate')
plt.hist(surf_plan_filtered['mb_used'], bins=20, alpha=0.5, label='surf')

# menambahkan judul dan nama sumbu
plt.xlabel('Jumlah Penggunaan Data', size=10)
plt.ylabel('Count', size=10)
plt.title('Overlay dari Jumlah penggunaan volume data yang dibuthkan pengguna perbulan pada kedua paket')
plt.legend(loc='upper right');
```


    
![png](output_198_0.png)
    



```python
# Tabel Histogram dari Kepadatan frekuensi Jumlah penggunaan volume data yang dibuthkan pengguna perbulan pada kedua paket

plt.hist([ultimate_plan_filtered['mb_used'], surf_plan_filtered['mb_used']], label = ['ultimate', 'surf'], density = True)

# menambahkan judul dan nama sumbu
plt.ylabel('Frekuensi Kepadatan')
plt.xlabel('jumlah Penggunaa Data')
plt.title('frekuensi kepadatan dari Jumlah penggunaan volume data')
plt.legend()
plt.show()
```


    
![png](output_199_0.png)
    


Pada histogram daru jumlah penggunaan data pengguna perbulan dapat dilihat bahwa paket surf lebih banyak frekuensi penggunaannya per bulan. namun paket ultimate lebih banyak jumlahnya dibanding paket surf.

##### Rata-rata Pendapatan yang diperoleh dari paket ultimate dan surf 


```python
# Rata-rata pendapatan dari pengguna kedua paket

ultimate_avg = ultimate_plan_filtered['revenue'].sum() / len(ultimate_plan_filtered['revenue'])
surf_avg = surf_plan_filtered['revenue'].sum() / len(surf_plan_filtered['revenue'])
diff = (ultimate_avg - surf_avg) / ultimate_avg * 100
print('Rata-rata pendapatan dari pengguna paket ultimate adalah :, ${:.2f}'.format(ultimate_avg))
print('Rata-rata pendapatan dari pengguna Paket Surf adalah : ${:.2f}'.format(surf_avg))
print('Persentasi Selisih Pendapatan paket ultimate and paket surf adalah {:.2f}%'.format(diff))
```

    Rata-rata pendapatan dari pengguna paket ultimate adalah :, $70.36
    Rata-rata pendapatan dari pengguna Paket Surf adalah : $47.71
    Persentasi Selisih Pendapatan paket ultimate and paket surf adalah 32.20%


Kesimpulan :

1. Setelah Menghitung distribusi statistik data, kita dapat mengetahui perilaku dari pengguna di kedua paket, dimana paket ultimate rata-rata pengguna melakukan 58 panggilan, dengan menghabiskan durasi panggilan 411 menit perbulan, dan mengirim pesan sekitar 32 perbulan ,menggunakan data 16645 MB data perbulan.
___________________________

2. Pada paket surf kita dapat melihat rata-rata pengguna melakukan panggilan sebanyak 56, dengan durasi 405 menit perbulan, dan mengirim pesan sebanyak 26, dan menggunakan data 15662 MB.

_____________________________

3. Dengan memplot histogram kepadatan frekuensi kita dapat mengamati bahwa pengguna paket surf lebih banyak memliki frekuensi secara keseluruhan dari pada pengguna paket ultimate. namun rata-rata pengguna paket ultimate memiliki banyak durasi panggilan perbulan, dan mengirim pesan lebih banyak lalu membutuhkan lebih banyak volume data dibandingkan dengan paket surf. 

_____________________________

4. Dapat ditarik kesimpulan bahwa paket ultimate banyak keuntungan pendapatan rata-rata dibandingkan pada paket surf.

## Menguji Hipotesis

#### Hipotesis 1

Test 1 

Hipotesis nol : Tidak ada perbedaan antara rata-rata pendapatan dari pengguna paket ultimate dan surf.

Setelah mengetahui perbandingan  pendapatan dari paket ultimate sebesar  usd70,32, dan paket  surf sebesar usd47.80, kita ingin mengetahui apakah perbedaan dari pendapatan ini signifikan? itu tergantung dari Variansnya dimana dari perhitungan sample dari nilai distribusinya. alih-alih berasumsi berdasarkan rata-rata saja untuk memastikan, kita menggunakan data untuk melakukan uji statistik, pada percobaan ini, hipotesis nol adalah tidak ada perbedaan rata-rata dari penggunaan paket ultimate dan surf. dana hipotesis alternatifnya adlaha bahwa pendapatan rata-rata dari pengguna paket ultimate dan surf berbeda. kita dapat melakukan pengujian hipoteisi ini dengan menggunakan tingkat signifikansi atau alpha 0.05 yang berarti dalam kasus ini 5% tingat kesalahannya. dimana kita kan menolaj hipotesis nol ketika hipotesis alternatif nya benar. lalu menggunakan uji-t untuk menguji hipotesis karena membandingkan rata-rata dua kelompok untuk menentukan apakah kedua kelompok ini berbeda satu sama lain.


1. Ho (Hipotesis nol)           = Rata-rata pendapatan pada paket Ultimate = (sama dengan)  Rata-rata pendapatan pada paket Surf
_______________________________
2. H1 (Hipotesis alternatif)      = Rata-rata pendapatan pada paket Ultimate <> (berbeda dengan) Rata-rata pendapatan pada paket Surf
___________________________________
3. α (alpha) tingkat signifikansi  = 0.05 
____________________________________
4. Jika p-value < (lebih kecil), maka hipotesis nol ditolak. jika p-value > (lebih besar) , maka hipotesis nol diterima.


```python
# Uji Hipotesis
ultimate = ultimate_plan_filtered['revenue']
surf = surf_plan_filtered['revenue']

# signifikasi alpha 
alpha = 0.05

# menguji hipotesis dimana rata-rata dari dua populasi bebas adalah sama
results = st.ttest_ind(ultimate, surf, equal_var = False )
print('Hasil p-value adalah:{}'.format(results.pvalue))

# Hasil dari uji hipotesis 
if (results.pvalue < alpha) :
    print('Kita dapat menolak Ho')
else :
    print('Kita tidak dapat menolak Ho')
```

    Hasil p-value adalah:3.0381763230534514e-124
    Kita dapat menolak Ho


Kesimpulan : 

Setelah menguji hipotesis nol dengan pernyataan bahwa tidak ada perbedaan antara rata-rata dari pendatapan penggunaan paket ultimate dan paket surf. dengan hasil uji p-value lebih kecil dari nilia alpha / signifikansi 0.05 dimana hipotesis nol ditolak. berarti analisi dari uji hipotesi menunjukkan baghwa rata-rata pendapatan penggunaan paket ultimate dan paket surf berbeda.

Test 2 

Hipotesis nol : tidak ada perbedaan rata-rata dari pengguna di NY-NJ dengan pendapatan pengguna dari wilayah lain. 

Hipotesis alternatif : terdapat perbedaan rata-rata pendapatan dari pengguna di wilaya NY-NJ dengan pendapatan pengguna dari wilayah lain. 
______________________
1. Pada tahap pengujian hipotesis diatas, kita akan menggunakan tingkat signifikansi atau alpha 0.05, lalu dengan menggunakan uji-t untuk menguji hipotesis untuk membandingkan rata-rata dua kelompok / sample, apakah dua kelompok ini berbeda satu sama lain.

1. Ho (Hipotesis nol) =  rata-rata pendapatan dari pengguna di NY-NJ == (sama dengan) rata-rata pendapatan dari pengguna dari wilayah lain.
_________________________________
2. H1 (Hipotesis alternatif) = rata-rata pendapatan dari pengguna di NY-NJ <> (berbeda dengan) rata-rata pendapatan dari pengguna dari wilayah lain.
_________________________________
3. α (alpha) tingkat signifikansi = 0.05
__________________________________
4. Jika p-value < (lebih kecil), maka hipotesis nol ditolak. jika p-value > (lebih besar) , maka hipotesis nol diterima.


```python
# Filter data untuk menguji hipotesis

# Filter pengguna ny-nj
newyork_jersey = agg_data.query('city == "New York-Newark-Jersey City, NY-NJ-PA MSA"')['revenue']

# filter pengguna not ny-nj
not_newyork_jersey = agg_data.query('city != "New York-Newark-Jersey City, NY-NJ-PA MSA"')['revenue']

# menampilkan rata-rata pendaptan dari pengguna ny-nj dan not-ny-nj
print('Rata-rata pendapatan dari pengguna NY-NJ adalah ${:.2f}'.format(newyork_jersey.mean()))
print('Rata-rata pendapatan dari pengguna selain NY-NJ adalah ${:.2f}'.format(not_newyork_jersey.mean()))
```

    Rata-rata pendapatan dari pengguna NY-NJ adalah $59.87
    Rata-rata pendapatan dari pengguna selain NY-NJ adalah $65.12


Setelah menentukan rata-rata pendapatan dari pengguna di wilayah NY-NJ sebesar USD59.92, dan rata-rata pendapatan pengguna dari wilayah lain sebesar USD65.22, apakah pendapatan pada dua pengguna ini signifikan? mari kita uji hipotesis dengan tingkat signifikansi 0.05, apakah kita dapat menolak hipotesis nol atau tidak. kita akan menggunakan uji-t untuk menguji hipotesis karena kita akan membandingkan rata-rata dua kelompok untuk menentukan apakah kedua kelompok berbeda satu sama lain. 


```python
# Uji Hipotesis
newyork_jersey = agg_data.query('city == "New York-Newark-Jersey City, NY-NJ-PA MSA"')['revenue']
not_newyork_jersey = agg_data.query('city != "New York-Newark-Jersey City, NY-NJ-PA MSA"')['revenue']

# signifikasi alpha 
alpha = 0.05

# menguji hipotesis dimana rata-rata dari dua populasi bebas adalah sama
results = st.ttest_ind(newyork_jersey, not_newyork_jersey, equal_var = False)
print('The p-value is: {}'.format(results.pvalue))

# Hasil dari uji hipotesis 
if (results.pvalue < alpha) :
    print('Kita dapat menolak Ho')
else :
    print('Kita tidak dapat menolak Ho')
```

    The p-value is: 0.03528813156344506
    Kita dapat menolak Ho


Kesimpulan : 

Setelah merumuskan hipotesis nol dengan asumsi bahwa tidak ada perbedaan antara rata-rata dari pengguna di wilayah NY-NJ dan pengguna dari wilayah lain. dengan menggunakan uji-t untuk menguji hipotesis menggunakan tingkat signifikasnsi 0.05. karena p-value lebih kecil dari alpha 0.05. dapat ditarik kesimpulan bahwa hipotesis nol dapat ditolak. namun dengan menghitung jumlah rata-rata pendapatan dari pengguna di kedua sample, kita dapat mengatakan bahwa rata-rata pendapatan di kedua wilayah memiliki nilai yang signifikan.

## Kesimpulan Akhir

1. Dengan melihat informasi umum dari data, kami mengidentifikasi beberapa kesalahan seperti masalah dengan tipe data dan memperbaikinya dengan mengubah tipe data ke format yang tepat. Kami menganalisis data dengan melakukan analisis data eksplorasi dan menemukan bahwa distribusi data sedikit miring ke kanan. Kami mendeteksi dan menghapus beberapa outlier dari data, dan menggunakan data yang difilter untuk menghitung statistik. Kami menentukan bahwa pengguna paket ulitmate  rata-rata melakukan 58 panggilan, menggunakan 411 menit per bulan, mengirim sekitar 32 pesan, dan menggunakan 16645 MB data per bulan. Kami juga menghitung statistik untuk pengguna surf, dan mengamati bahwa rata-rata pengguna melakukan 56 panggilan, menggunakan 405 menit per bulan, mengirim sekitar 26 pesan, dan menggunakan 15662 MB data per bulan.
_______________________________

2. dengan memplot distribusi statistik data dengan kepadatan frekuensi pekate per pengguna untuk mengata bhwa pengguna paket ultimate memiliki jumlah panggilan, durasi panggilan,  mengirimkan pesan lebih banyak, membutuhkan lebih banyak volume data dibandingkan dengan pengguna pada paket surf. lalu kita juga menentukan bahwa pengguna paket ultimate menghasilkan lebih banyak pendapatan rata-rata dengan selisih persentasi pendapatan sebesar 32.02%. Rata-rata pendapatan dari pengguna paket ultimate adalah USD70.32 dan Rata-rata pendapatan dari pengguna Paket Surf adalah USD47.80.
_________________________________

3. Setelah menguji hipotesis nol dengan pernyataan bahwa tidak ada perbedaan antara rata-rata dari pendatapan penggunaan paket ultimate dan paket surf. dengan hasil uji p-value lebih kecil dari nilia alpha / signifikansi 0.05 dimana hipotesis nol ditolak. berarti analisi dari uji hipotesi menunjukkan baghwa rata-rata pendapatan penggunaan paket ultimate dan paket surf berbeda, Setelah merumuskan hipotesis nol dengan asumsi bahwa tidak ada perbedaan antara rata-rata dari pengguna di wilayah NY-NJ dan pengguna dari wilayah lain. dengan menggunakan uji-t untuk menguji hipotesis menggunakan tingkat signifikasnsi 0.05. karena p-value lebih kecil dari alpha 0.05. dapat ditarik kesimpulan bahwa hipotesis nol dapat ditolak. namun dengan menghitung jumlah rata-rata pendapatan dari pengguna di kedua sample, kita dapat mengatakan bahwa rata-rata pendapatan di kedua wilayah memiliki nilai yang signifikan.

_________________________________

4. Dari analisa tersebut dapat kita simpulkan bahwa 

a. pengguna pada paket ultimate memiliki durasi panggilan lebih banyak perbulan, mengirim lebih banyak pesan teks, membutuhkan lebih banyak volume data dan menghasilkan pendapatan rata-rata lebih banyak daripada pengguna paket surf.

b. pengguna paket surf rata-rata melakukan lebih banyak panggulan daripada pengguna di paket ultimate.

c. pendapatan rata-rata dari pengguna di wilayah not NY-NJ lebih besar daripada pendapatan dari pengguna di wilayah NY-NJ.

e. departemen komersial harus berinvestasi dalam lebih banyak iklan di wilayah lain karena Megaline menghasilkan lebih banyak uang di wilayah itu daripada wilayah New York-New Jersey.

f. paket ultimate lebih menguntungkan daripada paket selancar meskipun paket selancar memiliki lebih banyak pengguna secara keseluruhan daripada paket surf.

g. Kita  dapat melakukan analisis lebih lanjut untuk menentukan pendapatan rata-rata berdasarkan kelompok usia. Itu juga akan menginformasikan kepada Megaline telekomunikasi tentang kelompok usia dan media periklanan mana yang akan ditargetkan untuk tujuan pemasaran. Misalnya, jika kita menganalisis bahwa orang-orang dalam kelompok usia 1 - 25 menghasilkan lebih banyak pendapatan, kita dapat menentukan apakah pemasaran di aplikasi media sosial seperti TikTok atau Instagram akan menghasilkan lebih banyak pengguna dan pendapatan daripada iklan TV atau papan reklame.



```python

```
