# Deskripsi Proyek

Anda bekerja di toko daring "Ice" yang menjual video game dari seluruh dunia. Data terkait ulasan pengguna dan ahli game, genre, platform (misalnya Xbox atau PlayStation), dan data historis penjualan game tersedia dari sumber terbuka. Anda perlu mengidentifikasi pola-pola yang menentukan apakah suatu game dapat dikatakan berhasil atau tidak. Dengan begitu, Anda bisa menemukan game yang paling berpotensial dan merencanakan kampanye iklannya.
_________________

Di depan Anda tersedia data dari tahun 2016. Mari bayangkan bahwa sekarang adalah bulan Desember tahun 2016 dan Anda sedang merencanakan kampanye untuk tahun 2017.
(Saat ini, yang terpenting bagi Anda adalah untuk mendapatkan pengalaman bekerja dengan data. Tidak masalah apakah Anda meramalkan penjualan tahun 2017 berdasarkan data dari tahun 2016 atau meramalkan penjualan tahun 2027 berdasarkan data dari tahun 2026.).
_________________
Dataset ini memuat singkatan. ESRB merupakan singkatan dari Entertainment Software Rating Board, yakni sebuah organisasi regulator mandiri yang mengevaluasi konten game dan memberikan rating usia seperti Remaja atau Dewasa.

## Memuat data dan mempelajari informasi keseluruhan pada data

### Memuat Libary yang dibutuhkan untuk pemrosesan data


```python
# Memuat semua library

# import pandas and numpy untuk proses dan manipulasi data
import pandas as pd
import numpy as np 
import random
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
plt.rcParams.update({'figure.max_open_warning': 0})

# tree map visualization
!pip install squarify
import squarify

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

    Collecting squarify
      Downloading squarify-0.4.3-py3-none-any.whl (4.3 kB)
    Installing collected packages: squarify
    Successfully installed squarify-0.4.3


### Memuat Data dari csv agar dapat dijalankan dengan pandas untuk menjadi DataFrame


```python
data = pd.read_csv('/datasets/games.csv')
```

###  Memuat Informasi dari dataset


```python
# Informasi dataset

print('Informasi dari dataset')
data
```

    Informasi dari dataset





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
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>NA_sales</th>
      <th>EU_sales</th>
      <th>JP_sales</th>
      <th>Other_sales</th>
      <th>Critic_Score</th>
      <th>User_Score</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>76.0</td>
      <td>8</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>15.68</td>
      <td>12.76</td>
      <td>3.79</td>
      <td>3.29</td>
      <td>82.0</td>
      <td>8.3</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>80.0</td>
      <td>8</td>
      <td>E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>16710</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016.0</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16711</th>
      <td>LMA Manager 2007</td>
      <td>X360</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16713</th>
      <td>Spirits &amp; Spells</td>
      <td>GBA</td>
      <td>2003.0</td>
      <td>Platform</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16714</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>16715 rows × 11 columns</p>
</div>




```python
print('informasi tipe data & panjang baris , column dari dataset')
data.info()
```

    informasi tipe data & panjang baris , column dari dataset
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16715 entries, 0 to 16714
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Name             16713 non-null  object 
     1   Platform         16715 non-null  object 
     2   Year_of_Release  16446 non-null  float64
     3   Genre            16713 non-null  object 
     4   NA_sales         16715 non-null  float64
     5   EU_sales         16715 non-null  float64
     6   JP_sales         16715 non-null  float64
     7   Other_sales      16715 non-null  float64
     8   Critic_Score     8137 non-null   float64
     9   User_Score       10014 non-null  object 
     10  Rating           9949 non-null   object 
    dtypes: float64(6), object(5)
    memory usage: 1.4+ MB



```python
print('distribusi statisik pada dataset')
data.describe()
```

    distribusi statisik pada dataset





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
      <th>Year_of_Release</th>
      <th>NA_sales</th>
      <th>EU_sales</th>
      <th>JP_sales</th>
      <th>Other_sales</th>
      <th>Critic_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16446.000000</td>
      <td>16715.000000</td>
      <td>16715.000000</td>
      <td>16715.000000</td>
      <td>16715.000000</td>
      <td>8137.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2006.484616</td>
      <td>0.263377</td>
      <td>0.145060</td>
      <td>0.077617</td>
      <td>0.047342</td>
      <td>68.967679</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.877050</td>
      <td>0.813604</td>
      <td>0.503339</td>
      <td>0.308853</td>
      <td>0.186731</td>
      <td>13.938165</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1980.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2003.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2007.000000</td>
      <td>0.080000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>71.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2010.000000</td>
      <td>0.240000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>41.360000</td>
      <td>28.960000</td>
      <td>10.220000</td>
      <td>10.570000</td>
      <td>98.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na')
data.isna().sum()
```

    memeriksa nilai yang hilang / na





    Name                  2
    Platform              0
    Year_of_Release     269
    Genre                 2
    NA_sales              0
    EU_sales              0
    JP_sales              0
    Other_sales           0
    Critic_Score       8578
    User_Score         6701
    Rating             6766
    dtype: int64




```python
print('panjang baris dari dataset :')
data.shape
```

    panjang baris dari dataset :





    (16715, 11)




```python
print('memeriksa duplikasi pada dataset :')
print('nilai duplikat pada dataset :', data.duplicated().sum())
```

    memeriksa duplikasi pada dataset :
    nilai duplikat pada dataset : 0



```python
# Fungsi untuk menghitung persentase dari column yang hilang 

def missing_values_table(data):
        # Total nilai yang hilang
        mis_val = data.isnull().sum()
        
        # Persentase nilai yang hilang
        mis_val_percent = 100 * data.isnull().sum() / len(data)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Merubah nama table
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Nilai yang hilang', 1 : '% Total Nilai dataset'})
        
        # Mengurutkan table berdasarkan persentase tertinggi
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% Total Nilai dataset', ascending=False).round(1)
        
        # Menampilkan beberapa informasi
        print ("Dataset yang dipilih " + str(data.shape[1]) + "  column.\n"      
            "Terdapat " + str(mis_val_table_ren_columns.shape[0]) +
              " column yang memiliki nilai yang hilang.")
        
        # Return dataset dengan nilai yang hilang
        return mis_val_table_ren_columns
```


```python
# Menampilkan hasil dari fungsi dataset dengan nilai yang hilang

missing_values_table(data)
```

    Dataset yang dipilih 11  column.
    Terdapat 6 column yang memiliki nilai yang hilang.





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
      <th>Nilai yang hilang</th>
      <th>% Total Nilai dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Critic_Score</th>
      <td>8578</td>
      <td>51.3</td>
    </tr>
    <tr>
      <th>Rating</th>
      <td>6766</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>User_Score</th>
      <td>6701</td>
      <td>40.1</td>
    </tr>
    <tr>
      <th>Year_of_Release</th>
      <td>269</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Genre</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Kesimpulan :

1. Dari hasil analisis awal pada dataset di dari 11 column terdapat 6 column yang memiliki nilai yang hilang : critic_score dengan jumlah 8578 / persentasi 51.3%, Rating dengan jumlah 6766 / persentasi 40.5%, User_Score dengan jumlah 6701 / persentasi 40.1%, Year_of_Release dengan jumlah 269 / persentasi 1.6%.
______________________

2. Selanjutnya kita akan mengubah tipe data dengan tipe data yang sesuai untuk langkah analisis selanjutnya, yakni pada column Year_of_Realese dan User_Score akan kita ubah menjadi tipe data interger.
______________________

3. Pada column Name & Genre teradapat nilai yang error dimana terdeteksi 2 value dengan nilai yang hilang pada baris columnya.
______________________

4. Pada Column Year_of_release terdapat nilai yang hilang secara acak (Missing at Random). & pada column Critic_Score, User_score dan Rating terdapat nilai hilang tidak secara acak (Missing not at Random).
______________________

5. Selanjutnya Kita juga perlu mengidentifikasi lebih detail mengenai nilai yang hilang pada column yang teridentifikasi. dan mengganti tipe data yang dibutuhkan. 

##  Data Preperation

### Mengubah nama column


```python
# Fungsi mengubah nama column
#def lowercase_columns(data):
   # return data.rename(str.lower, axis='columns')

#data = lowercase_columns(data)
#print(data.columns)    
```


```python
# merubah nama column pada dataset 

data.columns = data.columns.str.lower()

print(data.columns)
```

    Index(['name', 'platform', 'year_of_release', 'genre', 'na_sales', 'eu_sales',
           'jp_sales', 'other_sales', 'critic_score', 'user_score', 'rating'],
          dtype='object')


### Menangani nilai yang hilang


```python
# Menampilkan hasil dari fungsi dataset dengan nilai yang hilang

missing_values_table(data)
```

    Dataset yang dipilih 11  column.
    Terdapat 6 column yang memiliki nilai yang hilang.





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
      <th>Nilai yang hilang</th>
      <th>% Total Nilai dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>critic_score</th>
      <td>8578</td>
      <td>51.3</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>6766</td>
      <td>40.5</td>
    </tr>
    <tr>
      <th>user_score</th>
      <td>6701</td>
      <td>40.1</td>
    </tr>
    <tr>
      <th>year_of_release</th>
      <td>269</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>name</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>genre</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Mengatasi Nilai yang hilang pada column name & genre

Kita dapat mendrop nilai yang hilang pada column name dan genre, dikarenakan tidak ada cara lain untuk mengisi nilai yang hilang pada column name & genre baik menggunakan median atau mean pada value ini. dan juga karena persentasi nilai yang hilang sangat rendah maka kita bisa drop value nya tanpa menghilangkan analisis statistiknya.



```python
# Memfilter nilai yang hilang pada column name dan genre

data.loc[(data['name'].isna()) & (data['genre'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>659</th>
      <td>NaN</td>
      <td>GEN</td>
      <td>1993.0</td>
      <td>NaN</td>
      <td>1.78</td>
      <td>0.53</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14244</th>
      <td>NaN</td>
      <td>GEN</td>
      <td>1993.0</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop 'NaN' pada column name dan genre

data = data.dropna(subset= ['name', 'genre']).reset_index(drop=True)
```


```python
data[['name', 'genre']].isnull().sum()
```




    name     0
    genre    0
    dtype: int64



#### Mengatasi Nilai yang hilang year_of_release


```python
# Filter nilai yang hilang pada column years_of_release

data.loc[(data['year_of_release'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>183</th>
      <td>Madden NFL 2004</td>
      <td>PS2</td>
      <td>NaN</td>
      <td>Sports</td>
      <td>4.26</td>
      <td>0.26</td>
      <td>0.01</td>
      <td>0.71</td>
      <td>94.0</td>
      <td>8.5</td>
      <td>E</td>
    </tr>
    <tr>
      <th>377</th>
      <td>FIFA Soccer 2004</td>
      <td>PS2</td>
      <td>NaN</td>
      <td>Sports</td>
      <td>0.59</td>
      <td>2.36</td>
      <td>0.04</td>
      <td>0.51</td>
      <td>84.0</td>
      <td>6.4</td>
      <td>E</td>
    </tr>
    <tr>
      <th>456</th>
      <td>LEGO Batman: The Videogame</td>
      <td>Wii</td>
      <td>NaN</td>
      <td>Action</td>
      <td>1.80</td>
      <td>0.97</td>
      <td>0.00</td>
      <td>0.29</td>
      <td>74.0</td>
      <td>7.9</td>
      <td>E10+</td>
    </tr>
    <tr>
      <th>475</th>
      <td>wwe Smackdown vs. Raw 2006</td>
      <td>PS2</td>
      <td>NaN</td>
      <td>Fighting</td>
      <td>1.57</td>
      <td>1.02</td>
      <td>0.00</td>
      <td>0.41</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Space Invaders</td>
      <td>2600</td>
      <td>NaN</td>
      <td>Shooter</td>
      <td>2.36</td>
      <td>0.14</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>16371</th>
      <td>PDC World Championship Darts 2008</td>
      <td>PSP</td>
      <td>NaN</td>
      <td>Sports</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>43.0</td>
      <td>tbd</td>
      <td>E10+</td>
    </tr>
    <tr>
      <th>16403</th>
      <td>Freaky Flyers</td>
      <td>GC</td>
      <td>NaN</td>
      <td>Racing</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>69.0</td>
      <td>6.5</td>
      <td>T</td>
    </tr>
    <tr>
      <th>16446</th>
      <td>Inversion</td>
      <td>PC</td>
      <td>NaN</td>
      <td>Shooter</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>59.0</td>
      <td>6.7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>16456</th>
      <td>Hakuouki: Shinsengumi Kitan</td>
      <td>PS3</td>
      <td>NaN</td>
      <td>Adventure</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16520</th>
      <td>Virtua Quest</td>
      <td>GC</td>
      <td>NaN</td>
      <td>Role-Playing</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>55.0</td>
      <td>5.5</td>
      <td>T</td>
    </tr>
  </tbody>
</table>
<p>269 rows × 11 columns</p>
</div>



Untuk mengganti nilai yang hilang pada column years_of_release, pertama kita mencari nilai unik untuk mendapatkan daftar kemungkinan tahun rilis pada column nama game tersebut, kemudian memilih nilai acak dari daftar tersebut dan menetapkan ke tahun dimana game itu di rilis dalam dataframe. untuk nama yang unik dengan nilai yang hilang kita akan menggunakan fungsi mode() untuk mengisi nilai yang hilang pada kolom year_of_release. 


```python
# Fungsi untuk menetapkan nilai acak pada column year of release
def fill_in_year_of_release(data):
    # mendapatkan unik value dari nama games
    for name in data['name'].unique().tolist():
        # mendapatkan spesifik nama games untuk mengisi nilai nan di column year_of_release
        specific_name_data = data[data['name'] == name].dropna()['year_of_release']
        name_year_list = specific_name_data.unique().tolist()
        # looping untuk nilai yang hilang menetapkan nilai acak pada name untuk menjadi default value
        if name_year_list != []:
            data.loc[(data['name'] == name) & (data['year_of_release'] != data['year_of_release']), 'year_of_release'] = random.choice(name_year_list)
        else:
            data.loc[(data['name'] == name) & (data['year_of_release'] != data['year_of_release']), 'year_of_release'] = data['year_of_release'].mode()[0]
```


```python
# Menjalankan fungsi untuk nilai yang hilang pada column year_of_release

fill_in_year_of_release(data)
```


```python
# memfilter kembali nilai yang teridentifikasi terdapat nilai yang hilang

data.loc[(data['year_of_release'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Mengatasi Nilai yang hilang pada column critic_score 


```python
# Filter nilai yang hilang pada column critic_score

data.loc[(data['critic_score'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tetris</td>
      <td>GB</td>
      <td>1989.0</td>
      <td>Puzzle</td>
      <td>23.20</td>
      <td>2.26</td>
      <td>4.22</td>
      <td>0.58</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Duck Hunt</td>
      <td>NES</td>
      <td>1984.0</td>
      <td>Shooter</td>
      <td>26.93</td>
      <td>0.63</td>
      <td>0.28</td>
      <td>0.47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nintendogs</td>
      <td>DS</td>
      <td>2005.0</td>
      <td>Simulation</td>
      <td>9.05</td>
      <td>10.95</td>
      <td>1.93</td>
      <td>2.74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>16708</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016.0</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16709</th>
      <td>LMA Manager 2007</td>
      <td>X360</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16710</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16711</th>
      <td>Spirits &amp; Spells</td>
      <td>GBA</td>
      <td>2003.0</td>
      <td>Platform</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8576 rows × 11 columns</p>
</div>



Untuk mengisi nilai yang hilang pada colum critic score, kita akan mengambil nilai yang unik berdasarkan column platform untuk dijadikan value default dalam pemberian nilai pada platform game lalu mengisi nilai yang hilang tersebut dengan median dari penilian critic_score tersebut. 


```python
# Fungsi untuk menetapkan nilai acak pada column critic_score
def fill_in_critic_score(data):
    # mendapatkan unik value dari column platfrom
    for critic_score in data['platform'].unique().tolist():
        # mendapatkan nilai spesifik dari column platform
        specific_score_df = data[data['platform'] == critic_score].dropna()['critic_score']
        critic_score_list = specific_score_df.unique().tolist()
        # looping untuk nilai yang hilang menetapkan nilai acak pada platform untuk menjadi default value
        if critic_score_list != []:
            data.loc[(data['platform'] == critic_score) & (data['critic_score'] != data['critic_score']), 'critic_score'] = random.choice(critic_score_list)
        else:
            data.loc[(data['platform'] == critic_score) & (data['critic_score'] != data['critic_score']), 'critic_score'] = data['critic_score'].median()
```


```python
# Menjalankan fungsi untuk nilai yang hilang pada column critic score

fill_in_critic_score(data)
```


```python
# Filter nilai yang hilang pada column critic_score

data.loc[(data['critic_score'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Nilai yang hilang pada column user_score


```python
# Filter nilai yang hilang pada column user_score

data.loc[(data['user_score'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>69.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>69.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tetris</td>
      <td>GB</td>
      <td>1989.0</td>
      <td>Puzzle</td>
      <td>23.20</td>
      <td>2.26</td>
      <td>4.22</td>
      <td>0.58</td>
      <td>69.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Duck Hunt</td>
      <td>NES</td>
      <td>1984.0</td>
      <td>Shooter</td>
      <td>26.93</td>
      <td>0.63</td>
      <td>0.28</td>
      <td>0.47</td>
      <td>69.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nintendogs</td>
      <td>DS</td>
      <td>2005.0</td>
      <td>Simulation</td>
      <td>9.05</td>
      <td>10.95</td>
      <td>1.93</td>
      <td>2.74</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>16708</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016.0</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16709</th>
      <td>LMA Manager 2007</td>
      <td>X360</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>45.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16710</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16711</th>
      <td>Spirits &amp; Spells</td>
      <td>GBA</td>
      <td>2003.0</td>
      <td>Platform</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>73.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>6699 rows × 11 columns</p>
</div>



Untuk mengisi nilai yang hilang pada column user_score, kita akan mengambil nilai yang unik berdasarkan column platform untuk dijadikan value default dalam pemberian nilai pada platform game lalu mengisi nilai yang hilang tersebut dengan median dari penilian user_score tersebut. dan untuk nilai 'tbd' pada user score karena kemungkinan review pada games tersebut tidak ada yang memberikan insight maka nilai tbd akan di ganti menjadi nilai string 'NaN'. 


```python
# Fungsi untuk menetapkan nilai acak pada column user_score
def fill_in_user_score(data):
    data['user_score'] = data['user_score'].replace('tbd', np.nan)
    # mendapatkan unik value dari column platfrom
    for user_score in data['platform'].unique().tolist():
        # mendapatkan nilai spesifik dari column platform
        specific_score_data = data[data['platform'] == user_score].dropna()['user_score']
        user_score_list = specific_score_data.unique().tolist()
        # looping untuk nilai yang hilang menetapkan nilai acak pada platform untuk menjadi default value
        if user_score_list != []:
            data.loc[(data['platform'] == user_score) & (data['user_score'] != data['user_score']), 'user_score'] = random.choice(user_score_list)
        else:
            data.loc[(data['platform'] == user_score) & (data['user_score'] != data['user_score']), 'user_score'] = data['user_score'].median()   
```


```python
# Menjalankan fungsi untuk nilai yang hilang pada column critic score

fill_in_user_score(data)
```


```python
# Filter nilai yang hilang pada column user_score

data.loc[(data['user_score'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Mengatasi Nilai yang hilang pada column rating


```python
# Filter nilai yang hilang pada column user_score

data.loc[(data['rating'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>69.0</td>
      <td>7.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>69.0</td>
      <td>7.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tetris</td>
      <td>GB</td>
      <td>1989.0</td>
      <td>Puzzle</td>
      <td>23.20</td>
      <td>2.26</td>
      <td>4.22</td>
      <td>0.58</td>
      <td>69.0</td>
      <td>7.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Duck Hunt</td>
      <td>NES</td>
      <td>1984.0</td>
      <td>Shooter</td>
      <td>26.93</td>
      <td>0.63</td>
      <td>0.28</td>
      <td>0.47</td>
      <td>69.0</td>
      <td>7.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nintendogs</td>
      <td>DS</td>
      <td>2005.0</td>
      <td>Simulation</td>
      <td>9.05</td>
      <td>10.95</td>
      <td>1.93</td>
      <td>2.74</td>
      <td>48.0</td>
      <td>9.1</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>16708</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016.0</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>33.0</td>
      <td>8.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16709</th>
      <td>LMA Manager 2007</td>
      <td>X360</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>45.0</td>
      <td>1.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16710</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83.0</td>
      <td>5.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16711</th>
      <td>Spirits &amp; Spells</td>
      <td>GBA</td>
      <td>2003.0</td>
      <td>Platform</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>73.0</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83.0</td>
      <td>5.6</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>6764 rows × 11 columns</p>
</div>



Untuk mengatasi nilai yang hilang pada column rating kita akan mengambil nilai yang unik pada column platform untuk dijadikan value default dalam pemberian nilai pada platform game lalu mengisi nilai yang hilang dengan mode() dari penilaian rating berdasarkan column tersebut.


```python
# Fungsi untuk menetapkan nilai acak pada column rating
def fill_in_rating(data):
    # mendapatkan unik value dari column platfrom
    for rating in data['platform'].unique().tolist():
        specific_rating_data = data[data['platform'] == rating].dropna()['rating']
        rating_list = specific_rating_data.unique().tolist()
         # looping untuk nilai yang hilang menetapkan nilai acak pada platform untuk menjadi default value
        if rating_list != []:
            data.loc[(data['platform'] == rating) & (data['rating'] != data['rating']), 'rating'] = random.choice(rating_list)
        else:
            data.loc[(data['platform'] == rating) & (data['rating'] != data['rating']), 'rating'] = data['rating'].mode()[0]   
              
```


```python
# Menjalankan fungsi untuk nilai yang hilang pada column rating

fill_in_rating(data)
```


```python
# Filter nilai yang hilang pada column user_score

data.loc[(data['rating'].isna())]
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Melihat dataset setelah dimanipulasi 

data
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>76.0</td>
      <td>8</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>69.0</td>
      <td>7.6</td>
      <td>E</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>15.68</td>
      <td>12.76</td>
      <td>3.79</td>
      <td>3.29</td>
      <td>82.0</td>
      <td>8.3</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>80.0</td>
      <td>8</td>
      <td>E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>69.0</td>
      <td>7.6</td>
      <td>E</td>
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
    </tr>
    <tr>
      <th>16708</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016.0</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>33.0</td>
      <td>8.9</td>
      <td>E10+</td>
    </tr>
    <tr>
      <th>16709</th>
      <td>LMA Manager 2007</td>
      <td>X360</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>45.0</td>
      <td>1.2</td>
      <td>E</td>
    </tr>
    <tr>
      <th>16710</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83.0</td>
      <td>5.6</td>
      <td>M</td>
    </tr>
    <tr>
      <th>16711</th>
      <td>Spirits &amp; Spells</td>
      <td>GBA</td>
      <td>2003.0</td>
      <td>Platform</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>73.0</td>
      <td>3</td>
      <td>E</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016.0</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83.0</td>
      <td>5.6</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>16713 rows × 11 columns</p>
</div>




```python
# Menampilkan kembali persentase nilai yang hilang pada dataset

missing_values_table(data)
```

    Dataset yang dipilih 11  column.
    Terdapat 0 column yang memiliki nilai yang hilang.





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
      <th>Nilai yang hilang</th>
      <th>% Total Nilai dataset</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Kesimpulan :

Setelah memproses nilai yang hilang pada kolom name, genre, years_of_release, critic_score, user_score & rating, dimana kolom name, gender kita menghapus baris dengan nilai yang hilang dikarenakan nilai yang hilang kurang dari 1%, dan pada kolom years_of_release, critic_score, user_score & rating kita menerapkan fungsi khusu untuk mengisi nilai yang hilang dengan nilai acak dari nilai unik pada column name dan platform, dengan kondisi jika nilia unik kosong, maka nilai yang hilang diisi dengan median atau mode pada column yang terdapat nilai yang hilang serta singkatan dari 'tbd diubah menjadi string 'NaN' pada dataset. setelah memeriksa kembali nilai yang hilang dapat dilihat bahwa tidak ada nilai yang hilang kembali sehingga dataset siap untuk dianlisis. sebelum melanjutkan ke tahap analisis kita perlu mengkonversi tipe data ke tipe yang tepat sesuai dengan valuenya.


### Mengkonversi Tipe data 


```python
# Merubah tipe data ke tipe data yang sesuai

def convert_to_type(data, cols, type_val):
    for col in cols:
        data[col] = data[col].astype(type_val)
        
convert_to_type(data, ['name', 'platform', 'genre', 'rating'], str)
convert_to_type(data, ['year_of_release', 'critic_score'], int)
convert_to_type(data, ['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'user_score'], float)
```


```python
# Melihat informasi tipedata pada dataset

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16713 entries, 0 to 16712
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   name             16713 non-null  object 
     1   platform         16713 non-null  object 
     2   year_of_release  16713 non-null  int64  
     3   genre            16713 non-null  object 
     4   na_sales         16713 non-null  float64
     5   eu_sales         16713 non-null  float64
     6   jp_sales         16713 non-null  float64
     7   other_sales      16713 non-null  float64
     8   critic_score     16713 non-null  int64  
     9   user_score       16713 non-null  float64
     10  rating           16713 non-null  object 
    dtypes: float64(5), int64(2), object(4)
    memory usage: 1.4+ MB


Kita dapat melihat tipe data sudah diubah ke tipe data yang sesuai

### Menghitung Total penjualan di semua wilayah



```python
# Total penjualan di semua wilayah

data['total_sales'] = data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

```


```python
# Melihat total penjualan pada dataset

data
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006</td>
      <td>Sports</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>76</td>
      <td>8.0</td>
      <td>E</td>
      <td>82.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985</td>
      <td>Platform</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>69</td>
      <td>7.6</td>
      <td>E</td>
      <td>40.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008</td>
      <td>Racing</td>
      <td>15.68</td>
      <td>12.76</td>
      <td>3.79</td>
      <td>3.29</td>
      <td>82</td>
      <td>8.3</td>
      <td>E</td>
      <td>35.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009</td>
      <td>Sports</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>80</td>
      <td>8.0</td>
      <td>E</td>
      <td>32.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996</td>
      <td>Role-Playing</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>69</td>
      <td>7.6</td>
      <td>E</td>
      <td>31.38</td>
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
    </tr>
    <tr>
      <th>16708</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>33</td>
      <td>8.9</td>
      <td>E10+</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16709</th>
      <td>LMA Manager 2007</td>
      <td>X360</td>
      <td>2006</td>
      <td>Sports</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>45</td>
      <td>1.2</td>
      <td>E</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16710</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83</td>
      <td>5.6</td>
      <td>M</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16711</th>
      <td>Spirits &amp; Spells</td>
      <td>GBA</td>
      <td>2003</td>
      <td>Platform</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>73</td>
      <td>3.0</td>
      <td>E</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83</td>
      <td>5.6</td>
      <td>M</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
<p>16713 rows × 12 columns</p>
</div>



Kesimpulan :

Pada tahap ini kita telah mempersiapkan data dengan mengubah nama column menjadi lower case, kita mendrop nilai yang hilang pada column name dan genre, serta mengisi nilai yang hilang pada column years_of_release, critic_score, user_score & rating kita menerapkan fungsi khusu untuk mengisi nilai yang hilang dengan nilai acak dari nilai unik pada column name dan platform, dengan kondisi jika nilia unik kosong, maka nilai yang hilang diisi dengan median atau mode pada column yang terdapat nilai yang hilang serta singkatan dari 'tbd diubah menjadi string 'NaN' pada dataset, lalu merubah tipedata ke tipe data yang sesuai dan menghitung total penjualan pada seluruh wilayah. 


## Data Analyzing

### Banyaknya Game Yang Dirilis Pada Tahun Yang Berbeda 


```python
# Membuat dataset baru untuk dimanipulasi 

games_data = data.copy()
```


```python
# Mengelompokkan tahun game dirilis dan menghitung game yang dirilis

games_data_grouped = (games_data[['year_of_release', 'name']]
              .groupby('year_of_release')
              .agg('count')
              .sort_values('year_of_release')
              .rename(columns={'name':'games_count'}).reset_index()
)
```


```python
# Menampilkan dataset 

games_data_grouped
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
      <th>year_of_release</th>
      <th>games_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1982</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1984</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1985</td>
      <td>14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1986</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1987</td>
      <td>16</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1988</td>
      <td>15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1989</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1990</td>
      <td>16</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1991</td>
      <td>41</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1992</td>
      <td>43</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1993</td>
      <td>60</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1994</td>
      <td>121</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1995</td>
      <td>219</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1996</td>
      <td>263</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1997</td>
      <td>289</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1998</td>
      <td>379</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1999</td>
      <td>338</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2000</td>
      <td>350</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2001</td>
      <td>486</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2002</td>
      <td>843</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2003</td>
      <td>784</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2004</td>
      <td>764</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2005</td>
      <td>948</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2006</td>
      <td>1019</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007</td>
      <td>1201</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2008</td>
      <td>1607</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2009</td>
      <td>1430</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2010</td>
      <td>1266</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2011</td>
      <td>1144</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2012</td>
      <td>661</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2013</td>
      <td>547</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2014</td>
      <td>581</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2015</td>
      <td>606</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2016</td>
      <td>502</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fungsi untuk membuat plot lollipop chart

def plot_lollipop(data, x, y, title, ylabel):
    fig, ax=plt.subplots(figsize=(13,6))
    ax.vlines(x = data[x], ymin=0, ymax = data[y], color='purple', alpha=0.5, linewidth=2)
    ax.scatter(x = data[x], y= data[y], s= 75, color='blue', alpha=0.7)
    ax.set_title(title, fontdict={'size':12})
    ax.set_ylabel(ylabel, fontdict={'size':12})
    ax.set_xlabel(x, fontdict={'size':12})
    ax.set_xticks(data[x])
    ax.set_xticklabels(data[x], rotation=90, fontdict={'horizontalalignment':'right', 'size':10})    
    
```


```python
# Menjalankan fungsi plot 

plot_lollipop(games_data_grouped, 'year_of_release', 'games_count', 'Plot Chart Dari Game Yang Dirilis vs Tahun Game Dirilis', 'Banyak Game yang Dirilis')
```


    
![png](output_67_0.png)
    



```python
# Plot bar dari jumlah game yang dirilis vs tahun game dirilis

fig, ax = plt.subplots(figsize=(20,6))
ax = sns.barplot(data= games_data_grouped, x='year_of_release', y='games_count', edgecolor='black', hatch='/')
ax.set_title('Plot Chart Dari Game Yang Dirilis vs Tahun Game Dirilis', fontdict={'size':12})
ax.set_ylabel('Banyak Game yang Dirilis', fontdict={'size':12})
ax.set_xlabel('year_of_release', fontdict={'size':12})
ax.set_xticklabels(games_data_grouped['year_of_release'], rotation=90, fontdict={'horizontalalignment':'right', 'size':10});

# Value Label 
for index, row in games_data_grouped.iterrows():
    ax.text(row.name, row.games_count, round(row.games_count, 2), color='black', fontweight='bold' , ha="center")
```


    
![png](output_68_0.png)
    


Kesimpulan : 

Dapat dilihat dari plot lollipop plot diatas dimana game terbanyak dirilis pada tahun 2001 sd 2016. ini sangat masuk akal karena bertepatan dengan munculya web 2.0 dan internet pada dekadr pertama abad 21. dimana banyak game online, perangkat game seluler, layanan streaming sehingga banyak game dirilis karena orang dapat bermain game konsol di rumah. sebagian besar game dirilis antara tahun 2005 dan 2011, dan tahun dengan game dirilis terendah ada pada tahun 1980. penurunan jumlah game yang diliris mengalami penurunun setelah tahun 2008 kemungkinan dengan meningkatnya game seluler dari tahun 2008 dan seterusnya.

#### Data Pada Setiap Periode Tahun Game di Rilis 


```python
# Plot dari jumlah game berdasarkan rentang tahun dirilis 

bins = np.arange(games_data['year_of_release'].min(), games_data['year_of_release'].max()+1, 3)
bin_label = ['1980 - 1983', '1993 - 1986', '1986 - 1989', '1989 - 1992', '1992 - 1995', '1995 - 1998', '1998 - 2001', '2001 - 2004', '2004 - 2007', '2007 - 2010', '2010 - 2013', '2013 - 2016']
plot_bar = games_data.groupby(pd.cut(games_data['year_of_release'], bins=bins)).agg({'year_of_release': 'count'}).rename(columns={'year_of_release':'Number of games'})
plot_bar['Year range'] = bin_label
fig, ax=plt.subplots(figsize=(12,6))
ax = sns.barplot(x='Year range', y= 'Number of games', data = plot_bar, edgecolor='black', hatch='/')
ax.set_title('Plot Jumlah Periode Dari Tahun Game Dirilis', fontdict={'size':12})
ax.set_xticklabels(bin_label, rotation=45);

for i, v in enumerate(plot_bar.iloc[:,0].values):
    ax.text(i + 0.25, v + 3, str(v), color='black', fontweight='bold', fontdict={'horizontalalignment':'right', 'verticalalignment':'bottom', 'size':10})

```


    
![png](output_71_0.png)
    


Kesimpulan: 

Dari plot lollipop sebelumnya, kita dapat melihat bahwa tahun antara 2001 dan 2016 memiliki game yang paling banyak dirilis. Setelah itu kita mengelompokkan tahun berdasarkan rentang dan mencoba melihat seberapa signifikan game dirilis berdasarkan rentang periode game dirilis. disini kita membagi tahun dalam dua kelompok dengan interval tiga tahun. dan dapat diamati dari diagram batang bahwa untuk periode 2000 sd 2016 dengan jumlah game yang signifikan antara tahun 2007 - 2010 yang memiliki jumlah yang paling signifikan terbesar dalam data.


### Variasi Penjualan Dari Satu Platform ke Platform Lainnya 


```python
# Fungsi untuk membuat barplot

def plot_snsbar(data, x, y, title):
    xlabel = str(x.replace('_', ' ').capitalize())
    ylabel = str(y.replace('_', ' ').capitalize())
    # create grouped data
    data = data.groupby([x])[y].count().sort_values(ascending=False).reset_index()
    fig, ax=plt.subplots(figsize=(12,6))
    ax = sns.barplot(x = x, y = y, data=data)
    ax.set_title(title, fontdict={'size':12})
    ax.set_ylabel(ylabel, fontsize = 10)
    ax.set_xlabel(xlabel, fontsize = 10)
    ax.set_xticklabels(data[x], rotation=90);
```


```python
# Menjalankan Fungsi Barplot

# plot of sales variation from platform to platform
plot_snsbar(games_data, 'platform', 'total_sales', 'Plot Variasi Penjualan Dari Satu Platform Ke Platform Lainnya')
```


    
![png](output_75_0.png)
    



```python
platform_grouped = (games_data[['platform', 'total_sales']]
              .groupby('platform')
              .agg('count')
              .sort_values('total_sales').reset_index()
)
```


```python
platform_grouped
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
      <th>platform</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GG</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PCFX</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TG16</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3DO</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WS</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SCD</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NG</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GEN</td>
      <td>27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DC</td>
      <td>52</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GB</td>
      <td>98</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NES</td>
      <td>98</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2600</td>
      <td>133</td>
    </tr>
    <tr>
      <th>12</th>
      <td>WiiU</td>
      <td>147</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SAT</td>
      <td>173</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SNES</td>
      <td>239</td>
    </tr>
    <tr>
      <th>15</th>
      <td>XOne</td>
      <td>247</td>
    </tr>
    <tr>
      <th>16</th>
      <td>N64</td>
      <td>319</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PS4</td>
      <td>392</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PSV</td>
      <td>430</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3DS</td>
      <td>520</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GC</td>
      <td>556</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GBA</td>
      <td>822</td>
    </tr>
    <tr>
      <th>22</th>
      <td>XB</td>
      <td>824</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PC</td>
      <td>974</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PS</td>
      <td>1197</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PSP</td>
      <td>1209</td>
    </tr>
    <tr>
      <th>26</th>
      <td>X360</td>
      <td>1262</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Wii</td>
      <td>1320</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PS3</td>
      <td>1331</td>
    </tr>
    <tr>
      <th>29</th>
      <td>DS</td>
      <td>2151</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PS2</td>
      <td>2161</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot dari  Variasi Penjualan Dari Satu Platform ke Platform Lainnya
fig, ax = plt.subplots(figsize=(18,6))
ax = sns.barplot(data= platform_grouped , x='platform', y='total_sales', edgecolor='black', hatch='/', errwidth=0)
ax.set_title('Plot Variasi Penjualan Dari Satu Platform Ke Platform Lainnya', fontdict={'size':12})
ax.set_ylabel('Total Penjualan', fontdict={'size':12})
ax.set_xlabel('year_of_release', fontdict={'size':12})
ax.set_xticklabels(platform_grouped['platform'], rotation=90, fontdict={'horizontalalignment':'right', 'size':10});

# Value Label 
for index, row in platform_grouped.iterrows():
    ax.text(row.name, row.total_sales, round(row.total_sales, 2), color='black', fontweight='bold' , ha="center")
```


    
![png](output_78_0.png)
    


Kesimpulan :

Dapat Kita lihat dari barplot variasi dari penjualan platform game, dimana 5 teratas dari platform dengan penjualan terbanyak adalah PS2, DS, PS3, Wii dan X360. dan platfor dengan penjualan terendah adalah SCD, WS, 3DO, TG16, PCFX. Kita dapat memplot distribusi total penjulan berdasarkan tahun rilis dari setiap platform. ini akan memberikan kita gambaran mengenai platform populer dan durasu munculnya platform yang baru.

#### Memilih Platform Dengan Penjualan Terbesar dan Membuat Distribusi Berdasarkan Data PerTahun.



```python
# Fungsi Untuk Memplot distribusi chhart dari matplotlib

# function to plot distribution chart with matplotlib
def plot_distribution(data, x, y, column ='', value='', func=np.sum):
    if column != '' and value != '':
        filter_data = data[data[column] == value]
        plot_data = filter_data.pivot_table(index=x, values=y, aggfunc=func)
        values_to_plot = plot_data[y].values
    else:
        plot_data = data.pivot_table(index=x, values=y, aggfunc=func)
        values_to_plot = plot_data[y].values
    xlabel = x.replace('_', ' ').capitalize()
    ylabel = str(y.replace('_', ' ').capitalize())
    title = str(value) + " - " + ylabel + " vs. " + xlabel
    ax = plot_data.plot(kind='bar', figsize=(12,6), rot=45, title=title, edgecolor='silver', legend=False)
    ax.set_ylabel(ylabel, fontsize = 10)
    ax.set_xlabel(xlabel, fontsize = 10)
    ax.set_title(title, fontdict={'size':12}, fontweight='bold')
   
     # Mengatur Keterangan diatas bar 
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005), fontweight='bold', color='dodgerblue', horizontalalignment='left', size=10)
```


```python
# Distribusi Dari Total Sales Vs Tahun rilis berdasarkan setiap Platform 

for platform in data['platform'].unique():
    plot_distribution(data, 'year_of_release', 'total_sales', 'platform', platform)
```


    
![png](output_82_0.png)
    



    
![png](output_82_1.png)
    



    
![png](output_82_2.png)
    



    
![png](output_82_3.png)
    



    
![png](output_82_4.png)
    



    
![png](output_82_5.png)
    



    
![png](output_82_6.png)
    



    
![png](output_82_7.png)
    



    
![png](output_82_8.png)
    



    
![png](output_82_9.png)
    



    
![png](output_82_10.png)
    



    
![png](output_82_11.png)
    



    
![png](output_82_12.png)
    



    
![png](output_82_13.png)
    



    
![png](output_82_14.png)
    



    
![png](output_82_15.png)
    



    
![png](output_82_16.png)
    



    
![png](output_82_17.png)
    



    
![png](output_82_18.png)
    



    
![png](output_82_19.png)
    



    
![png](output_82_20.png)
    



    
![png](output_82_21.png)
    



    
![png](output_82_22.png)
    



    
![png](output_82_23.png)
    



    
![png](output_82_24.png)
    



    
![png](output_82_25.png)
    



    
![png](output_82_26.png)
    



    
![png](output_82_27.png)
    



    
![png](output_82_28.png)
    



    
![png](output_82_29.png)
    



    
![png](output_82_30.png)
    


Kesimpulan :

Pada Chart Distribusi Total Penjualan Terhadap Tahun Rilis Untuk Setiap Platform Diatas Dapat Disimpulkan :

1. Wii Adalah Console popular pada Tahun 2006 sd 2011. 
2. DS Adalah Console Populer antara tahun 2005 sd 2010.
3. X360 Adalah Console Populer antara tahun 2008 sd 2011.
4. PS2 Adalah Console Populer antara tahun 2001 dan 2007. Penjualan mulai menurun setelah tahun 2007 yang bertepatan dengan munculnya PS3.
5. PS3 Adalah Console Populer antara tahun 2008 sd 2013. dan penjualannya turun pada tahun 2016.
6. NES Adalah Console Populer pada tahun 1984 sd 1988.
7. PSP Populer pada  tahun 2005 sd 2010. 
8. PC Memiliki penjulan game tertinggi pada tahun 2011. PC Merupakan platform dengan umur terpanjang dibanding dengan platform lainnya. PC telah menjual Game selama 30 Tahun. 
9. Biasanya Dibutuhkan Sekitar 6 Tahun untuk muncul platform baru dan platform lama akan memudar.

#### Menemukan Platform yang dulunya populer tetapi sekarang tidak memiliki penjualan apapun

Kita dapat memvisualisasikan platform yang populer tetapi sekarang tidak memiliki penjualan berdasarkan z-score. kita dapat menghitung z-score dan menggunakan pendekatan bersyarat untuk membuat nilai penjualan yang kurang dari 0 dengan warna merah dan lebih dari 0 dengan warna hijau. kita dapat memvisualkannya dengan plot divergen. 


```python
# Mengelompokkan total penjualan berdasarkan platform games

platform_data = data[['platform', 'total_sales']].groupby('platform').sum().sort_values(by='total_sales').reset_index()
platform_data
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
      <th>platform</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PCFX</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GG</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3DO</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TG16</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WS</td>
      <td>1.42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NG</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SCD</td>
      <td>1.86</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DC</td>
      <td>15.95</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GEN</td>
      <td>28.35</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SAT</td>
      <td>33.59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PSV</td>
      <td>54.07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>WiiU</td>
      <td>82.19</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2600</td>
      <td>96.98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>XOne</td>
      <td>159.32</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GC</td>
      <td>198.93</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SNES</td>
      <td>200.04</td>
    </tr>
    <tr>
      <th>16</th>
      <td>N64</td>
      <td>218.68</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NES</td>
      <td>251.05</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GB</td>
      <td>255.46</td>
    </tr>
    <tr>
      <th>19</th>
      <td>XB</td>
      <td>257.74</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3DS</td>
      <td>259.00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PC</td>
      <td>259.52</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PSP</td>
      <td>294.05</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PS4</td>
      <td>314.14</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GBA</td>
      <td>317.85</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PS</td>
      <td>730.86</td>
    </tr>
    <tr>
      <th>26</th>
      <td>DS</td>
      <td>806.12</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Wii</td>
      <td>907.51</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PS3</td>
      <td>939.65</td>
    </tr>
    <tr>
      <th>29</th>
      <td>X360</td>
      <td>971.42</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PS2</td>
      <td>1255.77</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menghitung statistik distribusi zscore 

platform_data['sales_zscore'] = (platform_data['total_sales'] - platform_data['total_sales'].mean()) / platform_data['total_sales'].std()

```


```python
# Membuat perbedaan warna dari value penjualan platform

platform_data['color'] = ['red' if x < 0 else 'green' for x in platform_data['sales_zscore']]

```


```python
# Menampilkan Dataset 

platform_data
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
      <th>platform</th>
      <th>total_sales</th>
      <th>sales_zscore</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PCFX</td>
      <td>0.03</td>
      <td>-0.825614</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GG</td>
      <td>0.04</td>
      <td>-0.825586</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3DO</td>
      <td>0.10</td>
      <td>-0.825413</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TG16</td>
      <td>0.16</td>
      <td>-0.825241</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WS</td>
      <td>1.42</td>
      <td>-0.821623</td>
      <td>red</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NG</td>
      <td>1.44</td>
      <td>-0.821565</td>
      <td>red</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SCD</td>
      <td>1.86</td>
      <td>-0.820359</td>
      <td>red</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DC</td>
      <td>15.95</td>
      <td>-0.779896</td>
      <td>red</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GEN</td>
      <td>28.35</td>
      <td>-0.744287</td>
      <td>red</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SAT</td>
      <td>33.59</td>
      <td>-0.729239</td>
      <td>red</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PSV</td>
      <td>54.07</td>
      <td>-0.670425</td>
      <td>red</td>
    </tr>
    <tr>
      <th>11</th>
      <td>WiiU</td>
      <td>82.19</td>
      <td>-0.589672</td>
      <td>red</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2600</td>
      <td>96.98</td>
      <td>-0.547199</td>
      <td>red</td>
    </tr>
    <tr>
      <th>13</th>
      <td>XOne</td>
      <td>159.32</td>
      <td>-0.368174</td>
      <td>red</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GC</td>
      <td>198.93</td>
      <td>-0.254424</td>
      <td>red</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SNES</td>
      <td>200.04</td>
      <td>-0.251236</td>
      <td>red</td>
    </tr>
    <tr>
      <th>16</th>
      <td>N64</td>
      <td>218.68</td>
      <td>-0.197707</td>
      <td>red</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NES</td>
      <td>251.05</td>
      <td>-0.104748</td>
      <td>red</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GB</td>
      <td>255.46</td>
      <td>-0.092084</td>
      <td>red</td>
    </tr>
    <tr>
      <th>19</th>
      <td>XB</td>
      <td>257.74</td>
      <td>-0.085536</td>
      <td>red</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3DS</td>
      <td>259.00</td>
      <td>-0.081918</td>
      <td>red</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PC</td>
      <td>259.52</td>
      <td>-0.080425</td>
      <td>red</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PSP</td>
      <td>294.05</td>
      <td>0.018737</td>
      <td>green</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PS4</td>
      <td>314.14</td>
      <td>0.076430</td>
      <td>green</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GBA</td>
      <td>317.85</td>
      <td>0.087084</td>
      <td>green</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PS</td>
      <td>730.86</td>
      <td>1.273145</td>
      <td>green</td>
    </tr>
    <tr>
      <th>26</th>
      <td>DS</td>
      <td>806.12</td>
      <td>1.489273</td>
      <td>green</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Wii</td>
      <td>907.51</td>
      <td>1.780439</td>
      <td>green</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PS3</td>
      <td>939.65</td>
      <td>1.872737</td>
      <td>green</td>
    </tr>
    <tr>
      <th>29</th>
      <td>X360</td>
      <td>971.42</td>
      <td>1.963972</td>
      <td>green</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PS2</td>
      <td>1255.77</td>
      <td>2.780554</td>
      <td>green</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menampilkan Barplot Divergent dari nilai z-score total penjualan platform game

plt.figure(figsize=(14, 8))
plt.hlines(y=platform_data.platform, xmin=0, xmax=platform_data.sales_zscore, color=platform_data.color, alpha=0.4, linewidth=5)
plt.xlabel('z-score') 
plt.ylabel('Platform') 

# Menampilkan judul 
plt.title('Barplot Divergent dari total penjualan Platform')
plt.show()
```


    
![png](output_89_0.png)
    


Kesimpulan :

Pada barplot divergen diatas menunjukkan bar denga warna 'hijau adalah total penjualan platform dengan rata-rata terbanyak yakni PS2, x360, PS3, Wii, DS dan PS sedangkan bar dengan warna 'merah' adalah total penjualan dengan rata-rata terendah yakni TG16, 3DO, GG dan PCFX. 

### Menentukan Periode Waktu Pengambilan Data 


```python
# Menentukan Periode Waktu Pengambilan Data

new_data = data[data.year_of_release >= 2013]
new_data
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
      <th>name</th>
      <th>platform</th>
      <th>year_of_release</th>
      <th>genre</th>
      <th>na_sales</th>
      <th>eu_sales</th>
      <th>jp_sales</th>
      <th>other_sales</th>
      <th>critic_score</th>
      <th>user_score</th>
      <th>rating</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Grand Theft Auto V</td>
      <td>PS3</td>
      <td>2013</td>
      <td>Action</td>
      <td>7.02</td>
      <td>9.09</td>
      <td>0.98</td>
      <td>3.96</td>
      <td>97</td>
      <td>8.2</td>
      <td>M</td>
      <td>21.05</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Grand Theft Auto V</td>
      <td>X360</td>
      <td>2013</td>
      <td>Action</td>
      <td>9.66</td>
      <td>5.14</td>
      <td>0.06</td>
      <td>1.41</td>
      <td>97</td>
      <td>8.1</td>
      <td>M</td>
      <td>16.27</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>PS4</td>
      <td>2015</td>
      <td>Shooter</td>
      <td>6.03</td>
      <td>5.86</td>
      <td>0.36</td>
      <td>2.38</td>
      <td>90</td>
      <td>7.9</td>
      <td>M</td>
      <td>14.63</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Pokemon X/Pokemon Y</td>
      <td>3DS</td>
      <td>2013</td>
      <td>Role-Playing</td>
      <td>5.28</td>
      <td>4.19</td>
      <td>4.35</td>
      <td>0.78</td>
      <td>81</td>
      <td>4.4</td>
      <td>E10+</td>
      <td>14.60</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Grand Theft Auto V</td>
      <td>PS4</td>
      <td>2014</td>
      <td>Action</td>
      <td>3.96</td>
      <td>6.31</td>
      <td>0.38</td>
      <td>1.97</td>
      <td>97</td>
      <td>8.3</td>
      <td>M</td>
      <td>12.62</td>
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
    </tr>
    <tr>
      <th>16701</th>
      <td>Strawberry Nauts</td>
      <td>PSV</td>
      <td>2016</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83</td>
      <td>5.6</td>
      <td>M</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16705</th>
      <td>Aiyoku no Eustia</td>
      <td>PSV</td>
      <td>2014</td>
      <td>Misc</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83</td>
      <td>5.6</td>
      <td>M</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16708</th>
      <td>Samurai Warriors: Sanada Maru</td>
      <td>PS3</td>
      <td>2016</td>
      <td>Action</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>33</td>
      <td>8.9</td>
      <td>E10+</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16710</th>
      <td>Haitaka no Psychedelica</td>
      <td>PSV</td>
      <td>2016</td>
      <td>Adventure</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83</td>
      <td>5.6</td>
      <td>M</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>Winning Post 8 2016</td>
      <td>PSV</td>
      <td>2016</td>
      <td>Simulation</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>83</td>
      <td>5.6</td>
      <td>M</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
<p>2236 rows × 12 columns</p>
</div>



Kesimpulan :

Berdasarkan Data Pada Setiap Periode Tahun Game di Rilis disini kita menentukan periode waktu pengambilan data diatas tahun 2001.

### Platform Dengan Penjualan Terbanyak, Platform yang tumbuh lalu menyusut, Platform yang berpotensi menghasilkan keuntungan


```python
# Mengelompokkan platform dan total sale berdasarkan platform

tree_data = new_data[['platform', 'total_sales']].groupby('platform').sum().sort_values(by='total_sales').reset_index()

# filter dataset dengan total penjualan diatas 10%
tree_data = tree_data[tree_data['total_sales'] > 10]
tree_data
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
      <th>platform</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Wii</td>
      <td>13.66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PSV</td>
      <td>32.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PC</td>
      <td>39.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WiiU</td>
      <td>64.63</td>
    </tr>
    <tr>
      <th>6</th>
      <td>X360</td>
      <td>136.80</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3DS</td>
      <td>143.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>XOne</td>
      <td>159.32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PS3</td>
      <td>181.43</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PS4</td>
      <td>314.14</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Membuat label untuk tree map dari total penjualan platform

tree_size = tree_data['total_sales'].values.tolist()
labels = tree_data.apply(lambda x: str(x[0])+'\n'+'$'+ str(round(x[1])), axis=1)
labels

```




    2       Wii\n$14
    3       PSV\n$33
    4        PC\n$40
    5      WiiU\n$65
    6     X360\n$137
    7      3DS\n$143
    8     XOne\n$159
    9      PS3\n$181
    10     PS4\n$314
    dtype: object




```python
# Tree map dari total penjualan platform 

plt.figure(figsize=(14,10))
plt.title('Tree map dari total penjualan platform')
squarify.plot(sizes=tree_size, label=labels, alpha=0.5);
```


    
![png](output_97_0.png)
    


Kesimpulan : 

Dapat kita lihat pada ukuran dari tree map diatas, semua persegi empat mewakili skala dari penjualan yang tumbuh dan menyusut, total penjualan juga disertakan dalam tree map. ukuran pada seetiap persegi dari penjualan platform menginformasikan penjualan platform mana yang tumbuh atau menyusut. persegi yang lebih besar mewakili platform yang memimpin dalam penjualan, sedangkan persegi yang lebih kecil menggambarkan platform yang menyusut dalam penjualan. pada chart treemap diatas menunjukkan distribusi pasar dalam penjualan platform games dimana PS4, PS3, XOne, 3DS dan X360 adalah platform yang memimpin penjualan dan platform yang memiliki potensi untuk mendapatkan keuntungan . 

### Boxplot Penjualan Global Semua Game yang Dikelompokkan berdasarkan Platform


```python
# Mengelomppokka data untuk total penjualan game dari tahun 2013 keatas

grouped = new_data.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()
grouped
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
      <th>platform</th>
      <th>year_of_release</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3DS</td>
      <td>2013</td>
      <td>56.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3DS</td>
      <td>2014</td>
      <td>43.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3DS</td>
      <td>2015</td>
      <td>27.78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3DS</td>
      <td>2016</td>
      <td>15.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DS</td>
      <td>2013</td>
      <td>1.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PC</td>
      <td>2013</td>
      <td>12.66</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PC</td>
      <td>2014</td>
      <td>13.28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PC</td>
      <td>2015</td>
      <td>8.52</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PC</td>
      <td>2016</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PS3</td>
      <td>2013</td>
      <td>113.25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PS3</td>
      <td>2014</td>
      <td>47.76</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PS3</td>
      <td>2015</td>
      <td>16.82</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PS3</td>
      <td>2016</td>
      <td>3.60</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PS4</td>
      <td>2013</td>
      <td>25.99</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PS4</td>
      <td>2014</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PS4</td>
      <td>2015</td>
      <td>118.90</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PS4</td>
      <td>2016</td>
      <td>69.25</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PSP</td>
      <td>2013</td>
      <td>3.38</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PSP</td>
      <td>2014</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PSP</td>
      <td>2015</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PSV</td>
      <td>2013</td>
      <td>10.59</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PSV</td>
      <td>2014</td>
      <td>11.90</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PSV</td>
      <td>2015</td>
      <td>6.25</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PSV</td>
      <td>2016</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Wii</td>
      <td>2013</td>
      <td>8.59</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Wii</td>
      <td>2014</td>
      <td>3.75</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Wii</td>
      <td>2015</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Wii</td>
      <td>2016</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>28</th>
      <td>WiiU</td>
      <td>2013</td>
      <td>21.65</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WiiU</td>
      <td>2014</td>
      <td>22.03</td>
    </tr>
    <tr>
      <th>30</th>
      <td>WiiU</td>
      <td>2015</td>
      <td>16.35</td>
    </tr>
    <tr>
      <th>31</th>
      <td>WiiU</td>
      <td>2016</td>
      <td>4.60</td>
    </tr>
    <tr>
      <th>32</th>
      <td>X360</td>
      <td>2013</td>
      <td>88.58</td>
    </tr>
    <tr>
      <th>33</th>
      <td>X360</td>
      <td>2014</td>
      <td>34.74</td>
    </tr>
    <tr>
      <th>34</th>
      <td>X360</td>
      <td>2015</td>
      <td>11.96</td>
    </tr>
    <tr>
      <th>35</th>
      <td>X360</td>
      <td>2016</td>
      <td>1.52</td>
    </tr>
    <tr>
      <th>36</th>
      <td>XOne</td>
      <td>2013</td>
      <td>18.96</td>
    </tr>
    <tr>
      <th>37</th>
      <td>XOne</td>
      <td>2014</td>
      <td>54.07</td>
    </tr>
    <tr>
      <th>38</th>
      <td>XOne</td>
      <td>2015</td>
      <td>60.14</td>
    </tr>
    <tr>
      <th>39</th>
      <td>XOne</td>
      <td>2016</td>
      <td>26.15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mengelompokkan list berdasarkan platform

ordered = grouped.groupby(['platform'])['total_sales'].sum().sort_values().reset_index()['platform']
ordered
```




    0       DS
    1      PSP
    2      Wii
    3      PSV
    4       PC
    5     WiiU
    6     X360
    7      3DS
    8     XOne
    9      PS3
    10     PS4
    Name: platform, dtype: object



#### Signifikan Penjualan global dari platform


```python
# Boxplot dari Penjualan Global Semua Game yang Dikelompokkan berdasarkan Platform

plt.figure(figsize=(12,8))
sns.boxplot(x='platform', y='total_sales', data=grouped, order=ordered)
plt.title('Boxplot dari Penjualan Global Semua Game yang Dikelompokkan berdasarkan Platform')
plt.xlabel('platform')
plt.ylabel('total sales')
plt.show()
```


    
![png](output_103_0.png)
    


Kesimpulan :

Pada Boxplot diatas dapat dilihat bahwa platform PS4 memiliki rata-rata penjualan lebih tinggi dari tahun ke tahun dibandingkan dengan platform lainnya. dapat diamati bahwa perbedaan pada penjualan setiap platform sangat signifikan.dapat disimpulka bahwa PS4 memiliki penjualan lebih banyak daripada platform lainnya. kita juga dapat membuat boxplot yang dapat melihat rata-rata penjualan di seluruh platform.


```python
# Mencari rata-rata dari Penjualan Global Semua Game yang Dikelompokkan berdasarkan Platform

new_data_mean = new_data.groupby(['platform', 'year_of_release'])['total_sales'].agg('mean').reset_index()
new_data_mean
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
      <th>platform</th>
      <th>year_of_release</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3DS</td>
      <td>2013</td>
      <td>0.621648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3DS</td>
      <td>2014</td>
      <td>0.547000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3DS</td>
      <td>2015</td>
      <td>0.323023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3DS</td>
      <td>2016</td>
      <td>0.329130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DS</td>
      <td>2013</td>
      <td>0.192500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PC</td>
      <td>2013</td>
      <td>0.316500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PC</td>
      <td>2014</td>
      <td>0.282553</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PC</td>
      <td>2015</td>
      <td>0.170400</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PC</td>
      <td>2016</td>
      <td>0.097222</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PS3</td>
      <td>2013</td>
      <td>0.898810</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PS3</td>
      <td>2014</td>
      <td>0.442222</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PS3</td>
      <td>2015</td>
      <td>0.230411</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PS3</td>
      <td>2016</td>
      <td>0.094737</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PS4</td>
      <td>2013</td>
      <td>1.624375</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PS4</td>
      <td>2014</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PS4</td>
      <td>2015</td>
      <td>0.867883</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PS4</td>
      <td>2016</td>
      <td>0.422256</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PSP</td>
      <td>2013</td>
      <td>0.061455</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PSP</td>
      <td>2014</td>
      <td>0.024000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PSP</td>
      <td>2015</td>
      <td>0.040000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PSV</td>
      <td>2013</td>
      <td>0.168095</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PSV</td>
      <td>2014</td>
      <td>0.119000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PSV</td>
      <td>2015</td>
      <td>0.056818</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PSV</td>
      <td>2016</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Wii</td>
      <td>2013</td>
      <td>0.715833</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Wii</td>
      <td>2014</td>
      <td>0.625000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Wii</td>
      <td>2015</td>
      <td>0.285000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Wii</td>
      <td>2016</td>
      <td>0.180000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>WiiU</td>
      <td>2013</td>
      <td>0.515476</td>
    </tr>
    <tr>
      <th>29</th>
      <td>WiiU</td>
      <td>2014</td>
      <td>0.710645</td>
    </tr>
    <tr>
      <th>30</th>
      <td>WiiU</td>
      <td>2015</td>
      <td>0.583929</td>
    </tr>
    <tr>
      <th>31</th>
      <td>WiiU</td>
      <td>2016</td>
      <td>0.328571</td>
    </tr>
    <tr>
      <th>32</th>
      <td>X360</td>
      <td>2013</td>
      <td>1.181067</td>
    </tr>
    <tr>
      <th>33</th>
      <td>X360</td>
      <td>2014</td>
      <td>0.551429</td>
    </tr>
    <tr>
      <th>34</th>
      <td>X360</td>
      <td>2015</td>
      <td>0.341714</td>
    </tr>
    <tr>
      <th>35</th>
      <td>X360</td>
      <td>2016</td>
      <td>0.116923</td>
    </tr>
    <tr>
      <th>36</th>
      <td>XOne</td>
      <td>2013</td>
      <td>0.997895</td>
    </tr>
    <tr>
      <th>37</th>
      <td>XOne</td>
      <td>2014</td>
      <td>0.886393</td>
    </tr>
    <tr>
      <th>38</th>
      <td>XOne</td>
      <td>2015</td>
      <td>0.751750</td>
    </tr>
    <tr>
      <th>39</th>
      <td>XOne</td>
      <td>2016</td>
      <td>0.300575</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mengelompokkan list berdasarkan platform

ordered_mean = new_data_mean.groupby(['platform'])['total_sales'].sum().sort_values().reset_index()['platform']
ordered_mean
```




    0      PSP
    1       DS
    2      PSV
    3       PC
    4      PS3
    5      Wii
    6      3DS
    7     WiiU
    8     X360
    9     XOne
    10     PS4
    Name: platform, dtype: object



#### Rata-Rata Dari Penjualan Global Pada Platform


```python
# Boxplot dari Rata-rata Penjualan Global Semua Game yang Dikelompokkan berdasarkan Platform

plt.figure(figsize=(12,8))
sns.boxplot(x='platform', y='total_sales', data=new_data_mean, order=ordered_mean )
plt.title('Boxplot dari Rata-rata Penjualan Global Semua Game yang Dikelompokkan berdasarkan Platform')
plt.xlabel('platform')
plt.ylabel('total sales')
plt.show()
```


    
![png](output_108_0.png)
    


Kesimpulan : 

Dari boxplot rata-rata penjualan di berbagai platform, dapat dilihat bahwa PS4 memiliki rata-rata penjualan tertinggi dari platform lainnya, dan PS4 juga merupakan platform yang paling menguntungkan karena lebih banyak terjual.

### Ulasan  Pengguna dan para Profesional dalam mempengaruhi penjualan pada salah satu platform Populer. 

Fungsi Untuk mempermudah analisis :


```python
# Fungsi Menentukan Korelasi

def platform_corr(data, platform):
    data_platform = data[data.platform == platform].reset_index()[['user_score', 'critic_score', 'total_sales']]
    return data_platform.corr()
```


```python
# Fungsi menampilkan plot correlation matrix

def corrMatrix(data, platform):
    data_platform = data[data.platform == platform].reset_index()[['user_score', 'critic_score', 'total_sales']]
    plt.figure(figsize=(8, 6))
    corrMatrix = data_platform.corr()
    sns.heatmap(corrMatrix, annot=True, cmap='coolwarm')
    plt.title('Plot Correlation Matris ' + str(platform) + ' platform')
    plt.show();
```


```python
# Fungsi untuk menghitung korelasi pearson 

def pearson_coeff(data, x, y, platform):
    data_platform = data[data.platform == platform].reset_index()[['user_score', 'critic_score', 'total_sales']]
    pearson_coef, p_value = st.pearsonr(data_platform[x], data_platform[y])
    print("Hasil Korelasi Koefisin Pearson adalah {:.3f}".format(pearson_coef), "Dengan nilai p-value {:.3f}" .format(p_value))
    print()
    print("\033[1m" + 'Conclusion:' + "\033[0m")
    if (p_value < 0.001) and (pearson_coef < 0.5):
        print("Karena p-value adalah < 0.001, terdapat bukti kuat bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable tidak terlalu kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.001) and (pearson_coef < 0.7):
        print("Karena p-value adalah < 0.001, terdapat bukti kuat bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable cukup kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.001) and (pearson_coef > 0.7):
        print("Karena p-value adalah < 0.001, terdapat bukti kuat bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable sangat kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")") 
    elif (p_value < 0.05) and (pearson_coef < 0.5):
        print("Karena p-value adalah < 0.05,  terdapat bukti sedang bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika,  meskipun hubungan antar variable tidak terlalu kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.05) and (pearson_coef < 0.7):
        print("Karena p-value adalah < 0.05, terdapat bukti sedang bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable cukup kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.05) and (pearson_coef > 0.7):
        print("Karena p-value adalah < 0.05, terdapat bukti sedang bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable sangat kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.1) and (pearson_coef < 0.5):
        print("Karena p-value adalah < 0.1, terdapat bukti lemah bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable tidak terlalu kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.1) and (pearson_coef < 0.7):
        print("Karena p-value adalah < 0.1, terdapat bukti lemah bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable cukup kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")
    elif (p_value < 0.1) and (pearson_coef > 0.7):
        print("Karena p-value adalah < 0.1, terdapat bukti lemah bahwa korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika, meskipun hubungan antar variable sangat kuat ("+"\u2248"+"{:.3f}".format(abs(pearson_coef))+")")    
    elif (p_value > 0.1):
        print("Karena p-value adalah > 0.1, tidak terdapat bukti korelasi antara " + str(x.replace('_', ' ').capitalize()) + " and " + str(y.replace('_', ' ').capitalize()) + \
              " signifikan secara statistika")
```


```python
# Fungsi membuat scatter plot 

def plot_sns_scatter(data, x, y, platform):
    data_platform = data[data.platform == platform].reset_index()[[x, y]]
    plt.figure(figsize=(8,6))
    sns.regplot(x=x, y=y, data=data_platform)
    plt.title('scatter plot korelasi antara ' + str(x.replace('_', ' ').capitalize()) + \
              ' and ' + str(y.replace('_', '').capitalize()) + ' for ' + str(platform))
    plt.xlabel(str(x.replace('_', ' ').capitalize()))
    plt.ylabel(str(y.replace('_', ' ').capitalize()))
    plt.show();
    
```

Pada tahap ini, kita akan melihat bagaimana pengaruh Pengguna dan para Professional dalam mempengaruhi platform dari PS4, PS3, XOne, 3DS dan X360. Pertama kita akan melihat Korelasi PS4 dengan critic score, user score dan total sales.

#### PS4 Platform 


```python
# Korelasi PS4 terhadap Pengguna, para Professional terhadap penjualan 

platform_corr(new_data, 'PS4')
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
      <th>user_score</th>
      <th>critic_score</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_score</th>
      <td>1.000000</td>
      <td>0.631067</td>
      <td>-0.078504</td>
    </tr>
    <tr>
      <th>critic_score</th>
      <td>0.631067</td>
      <td>1.000000</td>
      <td>0.159895</td>
    </tr>
    <tr>
      <th>total_sales</th>
      <td>-0.078504</td>
      <td>0.159895</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot dari matrix korelasi

corrMatrix(new_data, 'PS4')
```


    
![png](output_119_0.png)
    



```python
# Fungsi Menghitung korelasi pearson antara pengguna terhadap total penjualan 

pearson_coeff(new_data, 'user_score', 'total_sales', 'PS4')
```

    Hasil Korelasi Koefisin Pearson adalah -0.079 Dengan nilai p-value 0.121
    
    [1mConclusion:[0m
    Karena p-value adalah > 0.1, tidak terdapat bukti korelasi antara User score and Total sales signifikan secara statistika



```python
# Membuat scatter plot untuk melihat korelasi antara pengguna terhadap penjualan

plot_sns_scatter(new_data, 'user_score', 'total_sales', 'PS4') 

```


    
![png](output_121_0.png)
    


Dapat dilihat pada scatter plot penggunga terhadap total penjualan, dimana titik data miring ke sebelah kanan, ini dapat diasumsikan bahwa korelasi pada dua variable lemah.


```python
# Fungsi Menghitung korelasi pearson antara critic score terhadap total penjualan 

pearson_coeff(new_data, 'critic_score', 'total_sales', 'PS4')
```

    Hasil Korelasi Koefisin Pearson adalah 0.160 Dengan nilai p-value 0.001
    
    [1mConclusion:[0m
    Karena p-value adalah < 0.05,  terdapat bukti sedang bahwa korelasi antara Critic score and Total sales signifikan secara statistika,  meskipun hubungan antar variable tidak terlalu kuat (≈0.160)



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'critic_score', 'total_sales', 'PS4') 

```


    
![png](output_124_0.png)
    


Dapat dilihat pada scatter plot critic score terhadap total penjualan, dimana titik data miring ke sebelah kiri, ini dapat diasumsikan bahwa korelasi pada dua variable ini kuat.

#### PS3 Platform 


```python
# Korelasi PS3 terhadap Pengguna, para Professional terhadap penjualan 

platform_corr(new_data, 'PS3')
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
      <th>user_score</th>
      <th>critic_score</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_score</th>
      <td>1.000000</td>
      <td>-0.188805</td>
      <td>-0.149824</td>
    </tr>
    <tr>
      <th>critic_score</th>
      <td>-0.188805</td>
      <td>1.000000</td>
      <td>0.321503</td>
    </tr>
    <tr>
      <th>total_sales</th>
      <td>-0.149824</td>
      <td>0.321503</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot dari matrix korelasi

corrMatrix(new_data, 'PS3')
```


    
![png](output_128_0.png)
    



```python
# Fungsi Menghitung korelasi pearson antara pengguna terhadap total penjualan 

pearson_coeff(new_data, 'user_score', 'total_sales', 'PS3')
```

    Hasil Korelasi Koefisin Pearson adalah -0.150 Dengan nilai p-value 0.005
    
    [1mConclusion:[0m
    Karena p-value adalah < 0.05,  terdapat bukti sedang bahwa korelasi antara User score and Total sales signifikan secara statistika,  meskipun hubungan antar variable tidak terlalu kuat (≈0.150)



```python
# Membuat scatter plot untuk melihat korelasi antara pengguna terhadap penjualan

plot_sns_scatter(new_data, 'user_score', 'total_sales', 'PS4') 
```


    
![png](output_130_0.png)
    


Dapat dilihat pada scatter plot penggunga terhadap total penjualan, dimana titik data miring ke sebelah kanan, ini dapat diasumsikan bahwa korelasi pada dua variable lemah.


```python
# Fungsi Menghitung korelasi pearson antara critic score terhadap total penjualan 

pearson_coeff(new_data, 'critic_score', 'total_sales', 'PS3')
```

    Hasil Korelasi Koefisin Pearson adalah 0.322 Dengan nilai p-value 0.000
    
    [1mConclusion:[0m
    Karena p-value adalah < 0.001, terdapat bukti kuat bahwa korelasi antara Critic score and Total sales signifikan secara statistika, meskipun hubungan antar variable tidak terlalu kuat (≈0.322)



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'critic_score', 'total_sales', 'PS3') 

```


    
![png](output_133_0.png)
    


Dapat dilihat pada scatter plot critic score terhadap total penjualan, dimana titik data miring ke sebelah kiri, ini dapat diasumsikan bahwa korelasi pada dua variable ini kuat.

#### XOne Platform 


```python
# Korelasi XOne terhadap Pengguna, para Professional terhadap penjualan 

platform_corr(new_data, 'XOne')
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
      <th>user_score</th>
      <th>critic_score</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_score</th>
      <td>1.000000</td>
      <td>0.575233</td>
      <td>0.027808</td>
    </tr>
    <tr>
      <th>critic_score</th>
      <td>0.575233</td>
      <td>1.000000</td>
      <td>0.379298</td>
    </tr>
    <tr>
      <th>total_sales</th>
      <td>0.027808</td>
      <td>0.379298</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot dari matrix korelasi

corrMatrix(new_data, 'XOne')
```


    
![png](output_137_0.png)
    



```python
# Fungsi Menghitung korelasi pearson antara pengguna terhadap total penjualan 

pearson_coeff(new_data, 'user_score', 'total_sales', 'XOne')
```

    Hasil Korelasi Koefisin Pearson adalah 0.028 Dengan nilai p-value 0.664
    
    [1mConclusion:[0m
    Karena p-value adalah > 0.1, tidak terdapat bukti korelasi antara User score and Total sales signifikan secara statistika



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'user_score', 'total_sales', 'XOne') 

```


    
![png](output_139_0.png)
    


Dapat dilihat pada scatter plot penggunga terhadap total penjualan, dimana titik data miring ke sebelah kanan, ini dapat diasumsikan bahwa korelasi pada dua variable lemah.


```python
# Fungsi Menghitung korelasi pearson antara critic score terhadap total penjualan 

pearson_coeff(new_data, 'critic_score', 'total_sales', 'XOne')
```

    Hasil Korelasi Koefisin Pearson adalah 0.379 Dengan nilai p-value 0.000
    
    [1mConclusion:[0m
    Karena p-value adalah < 0.001, terdapat bukti kuat bahwa korelasi antara Critic score and Total sales signifikan secara statistika, meskipun hubungan antar variable tidak terlalu kuat (≈0.379)



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'critic_score', 'total_sales', 'XOne') 

```


    
![png](output_142_0.png)
    


Dapat dilihat pada scatter plot critic score terhadap total penjualan, dimana titik data miring ke sebelah kiri, ini dapat diasumsikan bahwa terdapat korelasi pada dua variable ini walaupun tidak terlalu kuat.

#### 3DS Platform 


```python
# Korelasi 3DS terhadap Pengguna, para Professional terhadap penjualan 

platform_corr(new_data, '3DS')
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
      <th>user_score</th>
      <th>critic_score</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_score</th>
      <td>1.000000</td>
      <td>-0.213736</td>
      <td>0.067776</td>
    </tr>
    <tr>
      <th>critic_score</th>
      <td>-0.213736</td>
      <td>1.000000</td>
      <td>0.060622</td>
    </tr>
    <tr>
      <th>total_sales</th>
      <td>0.067776</td>
      <td>0.060622</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot dari matrix korelasi

corrMatrix(new_data, '3DS')
```


    
![png](output_146_0.png)
    



```python
# Fungsi Menghitung korelasi pearson antara pengguna terhadap total penjualan 

pearson_coeff(new_data, 'user_score', 'total_sales', '3DS')
```

    Hasil Korelasi Koefisin Pearson adalah 0.068 Dengan nilai p-value 0.239
    
    [1mConclusion:[0m
    Karena p-value adalah > 0.1, tidak terdapat bukti korelasi antara User score and Total sales signifikan secara statistika



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'user_score', 'total_sales', '3DS') 

```


    
![png](output_148_0.png)
    


Dapat dilihat pada scatter plot penggunga terhadap total penjualan, dimana titik data hampir sejajar, ini dapat diasumsikan bahwa tidak ada korelasi terhadap dua variable ini.


```python
# Fungsi Menghitung korelasi pearson antara critic score terhadap total penjualan 

pearson_coeff(new_data, 'critic_score', 'total_sales', '3DS')
```

    Hasil Korelasi Koefisin Pearson adalah 0.061 Dengan nilai p-value 0.293
    
    [1mConclusion:[0m
    Karena p-value adalah > 0.1, tidak terdapat bukti korelasi antara Critic score and Total sales signifikan secara statistika



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'critic_score', 'total_sales', '3DS') 

```


    
![png](output_151_0.png)
    


Dapat dilihat pada scatter plot critic score terhadap total penjualan, dimana titik data miring ke sebelah kiri, ini dapat diasumsikan bahwa terdapat korelasi pada dua variable ini walaupun tidak terlalu kuat.

#### X360 Platform 


```python
# Korelasi 3DS terhadap Pengguna, para Professional terhadap penjualan 

platform_corr(new_data, 'X360')
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
      <th>user_score</th>
      <th>critic_score</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_score</th>
      <td>1.000000</td>
      <td>0.516751</td>
      <td>0.063406</td>
    </tr>
    <tr>
      <th>critic_score</th>
      <td>0.516751</td>
      <td>1.000000</td>
      <td>0.322678</td>
    </tr>
    <tr>
      <th>total_sales</th>
      <td>0.063406</td>
      <td>0.322678</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot dari matrix korelasi

corrMatrix(new_data, 'X360')
```


    
![png](output_155_0.png)
    



```python
# Fungsi Menghitung korelasi pearson antara pengguna terhadap total penjualan 

pearson_coeff(new_data, 'user_score', 'total_sales', 'X360')
```

    Hasil Korelasi Koefisin Pearson adalah 0.063 Dengan nilai p-value 0.390
    
    [1mConclusion:[0m
    Karena p-value adalah > 0.1, tidak terdapat bukti korelasi antara User score and Total sales signifikan secara statistika



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'user_score', 'total_sales', 'X360') 

```


    
![png](output_157_0.png)
    


Dapat dilihat pada scatter plot penggunga terhadap total penjualan, dimana titik data hampir sejajar, ini dapat diasumsikan bahwa tidak ada korelasi terhadap dua variable ini.


```python
# Fungsi Menghitung korelasi pearson antara pengguna terhadap total penjualan 

pearson_coeff(new_data, 'critic_score', 'total_sales', 'X360')
```

    Hasil Korelasi Koefisin Pearson adalah 0.323 Dengan nilai p-value 0.000
    
    [1mConclusion:[0m
    Karena p-value adalah < 0.001, terdapat bukti kuat bahwa korelasi antara Critic score and Total sales signifikan secara statistika, meskipun hubungan antar variable tidak terlalu kuat (≈0.323)



```python
# Membuat scatter plot untuk melihat korelasi antara critic score terhadap penjualan

plot_sns_scatter(new_data, 'critic_score', 'total_sales', 'X360') 
```


    
![png](output_160_0.png)
    


Dapat dilihat pada scatter plot penggunga terhadap total penjualan, dimana titik data hampir sejajar, ini dapat diasumsikan bahwa tidak ada korelasi terhadap dua variable ini.

#### Kesimpulan :

Setelah menghitung p-value pada platform 3DS dan X360 tidak terdapat korelasi yang signifikan antara Ulasan Pengguna dan para Profesional terhadapa total penjualan. Lalu Platform (PS4, PS3 dan XOne) terdapat korelasi antara Ulasan Pengguna dan para Profesional terhadap total penjualan, dapat di simpulkan bahwa ulasa pengguna dan professional mempengaruhi total penjualan pada platform PS4, PS3 dan XOne.

### Distribusi umum game berdasarkan genre


```python
new_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2236 entries, 16 to 16712
    Data columns (total 12 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   name             2236 non-null   object 
     1   platform         2236 non-null   object 
     2   year_of_release  2236 non-null   int64  
     3   genre            2236 non-null   object 
     4   na_sales         2236 non-null   float64
     5   eu_sales         2236 non-null   float64
     6   jp_sales         2236 non-null   float64
     7   other_sales      2236 non-null   float64
     8   critic_score     2236 non-null   int64  
     9   user_score       2236 non-null   float64
     10  rating           2236 non-null   object 
     11  total_sales      2236 non-null   float64
    dtypes: float64(6), int64(2), object(4)
    memory usage: 227.1+ KB



```python
# Mengelompokkan genre terhadap total sales 

genre_grouped = new_data.groupby(['genre', 'year_of_release'])['total_sales'].sum().reset_index()
genre_grouped
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
      <th>genre</th>
      <th>year_of_release</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>2013</td>
      <td>122.79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Action</td>
      <td>2014</td>
      <td>97.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Action</td>
      <td>2015</td>
      <td>72.02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Action</td>
      <td>2016</td>
      <td>30.11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adventure</td>
      <td>2013</td>
      <td>6.09</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adventure</td>
      <td>2014</td>
      <td>5.57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adventure</td>
      <td>2015</td>
      <td>8.16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adventure</td>
      <td>2016</td>
      <td>3.82</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fighting</td>
      <td>2013</td>
      <td>7.09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fighting</td>
      <td>2014</td>
      <td>15.85</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Fighting</td>
      <td>2015</td>
      <td>7.90</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Fighting</td>
      <td>2016</td>
      <td>4.47</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Misc</td>
      <td>2013</td>
      <td>25.51</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Misc</td>
      <td>2014</td>
      <td>23.38</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Misc</td>
      <td>2015</td>
      <td>11.57</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Misc</td>
      <td>2016</td>
      <td>2.60</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platform</td>
      <td>2013</td>
      <td>24.54</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Platform</td>
      <td>2014</td>
      <td>8.81</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Platform</td>
      <td>2015</td>
      <td>6.05</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Platform</td>
      <td>2016</td>
      <td>3.23</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Puzzle</td>
      <td>2013</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Puzzle</td>
      <td>2014</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Puzzle</td>
      <td>2015</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Puzzle</td>
      <td>2016</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Racing</td>
      <td>2013</td>
      <td>12.37</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Racing</td>
      <td>2014</td>
      <td>16.66</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Racing</td>
      <td>2015</td>
      <td>8.07</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Racing</td>
      <td>2016</td>
      <td>2.79</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Role-Playing</td>
      <td>2013</td>
      <td>44.45</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Role-Playing</td>
      <td>2014</td>
      <td>45.62</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Role-Playing</td>
      <td>2015</td>
      <td>37.64</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Role-Playing</td>
      <td>2016</td>
      <td>18.18</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Shooter</td>
      <td>2013</td>
      <td>62.04</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Shooter</td>
      <td>2014</td>
      <td>65.21</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Shooter</td>
      <td>2015</td>
      <td>67.51</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Shooter</td>
      <td>2016</td>
      <td>38.22</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Simulation</td>
      <td>2013</td>
      <td>8.63</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Simulation</td>
      <td>2014</td>
      <td>5.58</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Simulation</td>
      <td>2015</td>
      <td>5.66</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Simulation</td>
      <td>2016</td>
      <td>1.89</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Sports</td>
      <td>2013</td>
      <td>41.17</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Sports</td>
      <td>2014</td>
      <td>45.15</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Sports</td>
      <td>2015</td>
      <td>40.84</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Sports</td>
      <td>2016</td>
      <td>23.49</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Strategy</td>
      <td>2013</td>
      <td>6.12</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Strategy</td>
      <td>2014</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Strategy</td>
      <td>2015</td>
      <td>1.85</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Strategy</td>
      <td>2016</td>
      <td>1.13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mengelompokkan list genre berdasarkan total sales

ordered_genre = genre_grouped.groupby(['genre'])['total_sales'].sum().sort_values().reset_index()['genre']
ordered_genre
```




    0           Puzzle
    1         Strategy
    2       Simulation
    3        Adventure
    4         Fighting
    5           Racing
    6         Platform
    7             Misc
    8     Role-Playing
    9           Sports
    10         Shooter
    11          Action
    Name: genre, dtype: object



<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>


```python
# Distribusi Statistik Genre terhadap total_sales

genre_grouped.describe()
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
      <th>year_of_release</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>48.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.500000</td>
      <td>22.733542</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.129865</td>
      <td>27.282646</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2013.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2013.750000</td>
      <td>4.307500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.500000</td>
      <td>8.720000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2015.250000</td>
      <td>37.785000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>122.790000</td>
    </tr>
  </tbody>
</table>
</div>



<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>


```python
# Boxplot dari total penjualan game berdasarkan genre
plt.figure(figsize=(12,8))
sns.boxplot(x='genre', y='total_sales', data=genre_grouped, order=ordered_genre)
plt.title('Boxplot dari total penjualan game berdasarkan genre  ')
plt.xlabel('Genre')
plt.ylabel('Total Sales');
```


    
![png](output_170_0.png)
    


<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

Dari boxplot diatas dapat dilihat bahwa genre yang paling menguntungkan adalah genre Action, Sport, dan Shooter. dan genre yang paling rendah keuntungannya adalah strategy dan puzzle. untuk melakukan generalisasi mengenai genre kita menghitung rata-rata dari total penjualan terhadap genre

#### Melakukan generalisasi terkait genre dengan menggunakan rata-rata total penjualan


```python
# Mengelompokkan rata-rata genre terhadap total sales 

mean_grouped_genre = new_data.groupby(['genre', 'year_of_release'])['total_sales'].agg('mean').reset_index()
mean_grouped_genre
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
      <th>genre</th>
      <th>year_of_release</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>2013</td>
      <td>0.824094</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Action</td>
      <td>2014</td>
      <td>0.517181</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Action</td>
      <td>2015</td>
      <td>0.284664</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Action</td>
      <td>2016</td>
      <td>0.169157</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adventure</td>
      <td>2013</td>
      <td>0.101500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adventure</td>
      <td>2014</td>
      <td>0.074267</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adventure</td>
      <td>2015</td>
      <td>0.151111</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adventure</td>
      <td>2016</td>
      <td>0.068214</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fighting</td>
      <td>2013</td>
      <td>0.354500</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fighting</td>
      <td>2014</td>
      <td>0.689130</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Fighting</td>
      <td>2015</td>
      <td>0.376190</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Fighting</td>
      <td>2016</td>
      <td>0.279375</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Misc</td>
      <td>2013</td>
      <td>0.593256</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Misc</td>
      <td>2014</td>
      <td>0.556667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Misc</td>
      <td>2015</td>
      <td>0.296667</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Misc</td>
      <td>2016</td>
      <td>0.081250</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Platform</td>
      <td>2013</td>
      <td>0.681667</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Platform</td>
      <td>2014</td>
      <td>0.881000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Platform</td>
      <td>2015</td>
      <td>0.465385</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Platform</td>
      <td>2016</td>
      <td>0.215333</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Puzzle</td>
      <td>2013</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Puzzle</td>
      <td>2014</td>
      <td>0.212857</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Puzzle</td>
      <td>2015</td>
      <td>0.118333</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Puzzle</td>
      <td>2016</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Racing</td>
      <td>2013</td>
      <td>0.773125</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Racing</td>
      <td>2014</td>
      <td>0.617037</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Racing</td>
      <td>2015</td>
      <td>0.448333</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Racing</td>
      <td>2016</td>
      <td>0.116250</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Role-Playing</td>
      <td>2013</td>
      <td>0.626056</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Role-Playing</td>
      <td>2014</td>
      <td>0.512584</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Role-Playing</td>
      <td>2015</td>
      <td>0.482564</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Role-Playing</td>
      <td>2016</td>
      <td>0.336667</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Shooter</td>
      <td>2013</td>
      <td>1.051525</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Shooter</td>
      <td>2014</td>
      <td>1.387447</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Shooter</td>
      <td>2015</td>
      <td>1.985588</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Shooter</td>
      <td>2016</td>
      <td>0.813191</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Simulation</td>
      <td>2013</td>
      <td>0.479444</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Simulation</td>
      <td>2014</td>
      <td>0.507273</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Simulation</td>
      <td>2015</td>
      <td>0.377333</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Simulation</td>
      <td>2016</td>
      <td>0.105000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Sports</td>
      <td>2013</td>
      <td>0.776792</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Sports</td>
      <td>2014</td>
      <td>0.836111</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Sports</td>
      <td>2015</td>
      <td>0.692203</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Sports</td>
      <td>2016</td>
      <td>0.489375</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Strategy</td>
      <td>2013</td>
      <td>0.322105</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Strategy</td>
      <td>2014</td>
      <td>0.122500</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Strategy</td>
      <td>2015</td>
      <td>0.115625</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Strategy</td>
      <td>2016</td>
      <td>0.086923</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mengelompokkan list genre berdasarkan total sales

ordered_genre_mean = mean_grouped_genre.groupby(['genre'])['total_sales'].sum().sort_values().reset_index()['genre']
ordered_genre_mean
```




    0        Adventure
    1         Strategy
    2           Puzzle
    3       Simulation
    4             Misc
    5         Fighting
    6           Action
    7           Racing
    8     Role-Playing
    9         Platform
    10          Sports
    11         Shooter
    Name: genre, dtype: object




```python
# Distribusi Statistik Genre terhadap total_sales

mean_grouped_genre.describe()
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
      <th>year_of_release</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>48.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.500000</td>
      <td>0.466309</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.129865</td>
      <td>0.372862</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2013.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2013.750000</td>
      <td>0.164646</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.500000</td>
      <td>0.412833</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2015.250000</td>
      <td>0.639959</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>1.985588</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Boxplot dari rata-rata total penjualan game berdasarkan genre
plt.figure(figsize=(12,8))
sns.boxplot(x='genre', y='total_sales', data=mean_grouped_genre, order=ordered_genre_mean)
plt.title('Boxplot dari rata-rata total penjualan game berdasarkan genre  ')
plt.xlabel('Genre')
plt.ylabel('Total Sales');
```


    
![png](output_177_0.png)
    


Dapat dilihat pada boxplot rata-rata total penjualan berdasarkan game pada seluruh genre. dapat disimpulkan bahwa penjualan dari game dengan genre petualangan dan strategy yang paling tidak menguntungkan. lalu untuk game dengan total penjualan yang menguntungkan adaah shooter, sport. 

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

#### Kesimpulan :

Setelah menganalisa, berapa banyak game yang dirilis pada tahun yang berbeda bahwa lebih banyak game dirilis pada tahun 2001 hingga 2016. Sebagian besar game dirilis antara tahun 2005 dan 2011. Tahun dengan jumlah game tertinggi yang dirilis adalah tahun 2008. Dari analisis bahwa periode 2000 hingga 2016 signifikan dengan tahun antara 2007 dan 2010 memiliki signifikansi paling besar dalam data. Analisis variasi penjualan di seluruh platform menunjukkan bahwa lima platform teratas dalam hal total penjualan masing-masing adalah PS4, PS3, XOne, 3DS dan X360. Platform dengan penjualan paling sedikit adalah PCFX, GG, 3DO, TG16, dan WS. kita juga membuktikan bahwa PC memiliki penjualan tertinggi pada tahun 2011. PC adalah platform dengan umur terpanjang di antara penjualan platform lainnya selama sekitar 30 tahun. Biasanya dibutuhkan sekitar 6 tahun untuk platform baru muncul dan yang lama memudar.

Kami menetapkan bahwa PS4, PS3, XOne, 3DS, dan X360 adalah platform yang memimpin dalam penjualan. Ini menjadikan mereka platform yang paling menguntungkan. PCFX, GG, 3DO, TG16, dan WS adalah platform terburuk dalam hal total penjualan dengan nilai jauh di bawah rata-rata data. dapat  dilihat bagaimana ulasan pengguna dan profesional memengaruhi penjualan untuk satu platform populer dari tahun 2013. Kami menyimpulkan setelah menghitung nilai-p dan menganalisis statistik bahwa ada hubungan linier yang signifikan antara ulasan pengguna dan profesional dan Total penjualan untuk produk teratas. Oleh karena itu, review pengguna mempengaruhi total penjualan.

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

##  Melakukan pemprofilan pengguna untuk masing-masing wilayah

Untuk setiap wilayah (NA, EU, JP), tentukan:
1. 5 platform teratas. Jelaskan variasi pangsa pasar dari satu wilayah ke wilayah lainnya.
1. 5 genre teratas. Jelaskan perbedaannya.
1. Apakah rating ESRB memengaruhi penjualan di masing-masing wilayah?

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

### Menentukan Penjulan berdasarkan Platform di setiap wilayah

#### Mengelompokkan Penjualan pada wilayah NA


```python
# Mengelompokkan Wilayah NA 

new_data_region = new_data.groupby(['platform'])['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending = False)
top_5_platforms = new_data_region.head()
(top_5_platforms.set_index('platform')
                .plot(y='na_sales', kind='pie',
                     title= 'Pie Chart Variasi Pangsa Pasar dari 5 Platform Teratas di Wilayah NA',
                     figsize=(8,8), autopct='%1.1f%%', shadow=True)

);

```


    
![png](output_187_0.png)
    


#### Mengelompokkan Penjualan pada wilayah EU


```python
new_data_region = new_data.groupby(['platform'])['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending = False)
top_5_platforms = new_data_region.head()
(top_5_platforms.set_index('platform')
                .plot(y='eu_sales', kind='pie',
                     title= 'Pie Chart Variasi Pangsa Pasar dari 5 Platform Teratas di Wilayah EU',
                     figsize=(8,8), autopct='%1.1f%%', shadow=True)

);
```


    
![png](output_189_0.png)
    


#### Mengelompokkan Penjualan pada wilayah JP


```python
new_data_region = new_data.groupby(['platform'])['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending = False)
top_5_platforms = new_data_region.head()
(top_5_platforms.set_index('platform')
                .plot(y='jp_sales', kind='pie',
                     title= 'Pie Chart Variasi Pangsa Pasar dari 5 Platform Teratas di Wilayah JP',
                     figsize=(8,8), autopct='%1.1f%%', shadow=True)

);
```


    
![png](output_191_0.png)
    


<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

### Menentukan Penjualan Berdasarkan Genre disetiap Wilayah

#### Mengelompokkan Penjualan Berdasarkan Genre di Wilayah NA


```python
# mengelompokkan penjualan berdasarkan genre di wilayah NA
new_data_genre = new_data.groupby(['genre'])['na_sales'].sum().reset_index().sort_values(by= 'na_sales', ascending = False )
top_5_genre_na = new_data_genre
top_5_genre_na.reset_index(drop=True)
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
      <th>genre</th>
      <th>na_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>126.07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shooter</td>
      <td>109.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sports</td>
      <td>65.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Role-Playing</td>
      <td>46.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Misc</td>
      <td>27.49</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Platform</td>
      <td>18.14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fighting</td>
      <td>15.55</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Racing</td>
      <td>12.96</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adventure</td>
      <td>7.14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Simulation</td>
      <td>4.86</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Strategy</td>
      <td>3.28</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Puzzle</td>
      <td>0.83</td>
    </tr>
  </tbody>
</table>
</div>



#### Mengelompokkan Penjualan Berdasarkan Genre di Wilayah EU


```python
# mengelompokkan penjualan berdasarkan genre di wilayah NA
new_data_genre = new_data.groupby(['genre'])['eu_sales'].sum().reset_index().sort_values(by= 'eu_sales', ascending = False )
top_5_genre_eu = new_data_genre
top_5_genre_eu.reset_index(drop=True)
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
      <th>genre</th>
      <th>eu_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>118.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shooter</td>
      <td>87.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sports</td>
      <td>60.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Role-Playing</td>
      <td>36.97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Racing</td>
      <td>20.19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Misc</td>
      <td>20.04</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Platform</td>
      <td>15.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Simulation</td>
      <td>10.92</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fighting</td>
      <td>8.55</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Adventure</td>
      <td>8.25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Strategy</td>
      <td>4.22</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Puzzle</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



#### Mengelompokkan Penjualan Berdasarkan Genre di Wilayah JP


```python
# mengelompokkan penjualan berdasarkan genre di wilayah NA
new_data_genre = new_data.groupby(['genre'])['jp_sales'].sum().reset_index().sort_values(by= 'jp_sales', ascending = False )
top_5_genre_jp = new_data_genre
top_5_genre_jp.reset_index(drop=True)
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
      <th>genre</th>
      <th>jp_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Role-Playing</td>
      <td>51.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Action</td>
      <td>40.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Misc</td>
      <td>9.44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fighting</td>
      <td>7.65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shooter</td>
      <td>6.61</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adventure</td>
      <td>5.82</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sports</td>
      <td>5.41</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Platform</td>
      <td>4.79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Simulation</td>
      <td>4.52</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Racing</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Strategy</td>
      <td>1.77</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Puzzle</td>
      <td>1.18</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

Penjualan genre games pada wilayah NA dan EU hampir mirip dimana genre action, shooter, sport mendominasi penjualan, dan pada wilayah JP penjualan yang mendominasi adalah role-playing, action dan misc. 

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

### Apakah rating ESRB ( Entertainment Software Rating Board ) memengaruhi penjualan di masing-masing wilayah?

#### Mengelompokkan Penjualan Berdasarkan rating di Wilayah NA


```python
# Mengelompokkan ESRB rating pada wilayah  NA
new_data_rating_na = new_data.groupby(['rating'])['na_sales'].sum().reset_index().sort_values(by='na_sales', ascending= False)
new_data_rating_na = new_data_rating_na.reset_index(drop=True)
new_data_rating_na
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
      <th>rating</th>
      <th>na_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>190.28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E10+</td>
      <td>105.65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E</td>
      <td>91.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>T</td>
      <td>49.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K-A</td>
      <td>0.90</td>
    </tr>
  </tbody>
</table>
</div>



#### Mengelompokkan Penjualan Berdasarkan rating di Wilayah EU


```python
# Mengelompokkan ESRB rating pada wilayah EU
new_data_rating_eu = new_data.groupby(['rating'])['eu_sales'].sum().reset_index().sort_values(by='eu_sales', ascending= False)
new_data_rating_eu = new_data_rating_eu.reset_index(drop=True)
new_data_rating_eu
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
      <th>rating</th>
      <th>eu_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>175.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E</td>
      <td>89.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E10+</td>
      <td>81.90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>T</td>
      <td>41.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K-A</td>
      <td>4.02</td>
    </tr>
  </tbody>
</table>
</div>



#### Mengelompokkan Penjualan Berdasarkan rating di Wilayah JP


```python
# Mengelompokkan ESRB rating pada wilayah JP
new_data_rating_jp = new_data.groupby(['rating'])['jp_sales'].sum().reset_index().sort_values(by='jp_sales', ascending= False)
new_data_rating_jp = new_data_rating_jp.reset_index(drop=True)
new_data_rating_jp
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
      <th>rating</th>
      <th>jp_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E10+</td>
      <td>69.87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>35.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>T</td>
      <td>20.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E</td>
      <td>15.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K-A</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

Melihat hasilnya, peringkat ESRB memang memengaruhi penjualan di masing-masing wilayah. Di wilayah na dan eu tersebut, rating M (MATURE) mendapat penjualan tertinggi. sedangkan di wilayah JP rating T (TEEN) dengan penjualan tertinggi.

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

#### Kesimpulan : 

Setelah melakukan analisis dari variasi dalam pangsa pasar pada lima platform dari setiap wilayah dapat disimpulkan, 

1. Wilayah NA dan EU  memiliki platform PS4 dengan pangsa pasar paling banyak yakni 28% dan 42% , sedangkan pada wilayah JP memiliki 3DS dengan 49%. 
_____________________________

2. Untuk penjualan dengan genre games pada wilayah NA dan UE memiliki genre Action dengan pangsa pasar terbanyak dengan jumlah 126 di wilayah NA dan EU 118, sedangkan pada WIlayah JP Role-Playing memiliki jumlah penjualan dengan jumlah 51. 
_____________________________

3. Untuk rating ESRB ( Entertainment Software Rating Board ), kita menemukan bahwa ESRB mempengaruhi penjualan di masing-masing wilayah. dimana Di wilayah na dan eu tersebut, rating M (MATURE) mendapat penjualan tertinggi. sedangkan di wilayah JP rating T (TEEN) dengan penjualan tertinggi.


<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

## Menguji hipotesis

Membuat fungsi untuk menentukan hasil Hipotesis 


```python
# Fungsi untuk menentukan hasil hipotesis

def hipotesis_test(variable1, variable2, alpha):
    
    # Menentukan nilai signifikasi alpha 
    alpha = alpha
    
    # Menguji hipotesis dengan rata-rata dari dua populasi bebas adalah sama
    results = st.ttest_ind(variable1, variable2, equal_var = False)
    print('Nilai p-value adalah : {}'.format(results.pvalue))
    
    # Hasil dari p-value
    if (results.pvalue < alpha ):
         print('Kita dapat menolak Ho')
    else :
        print('Kita tidak dapat menolak Ho')
```

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

### Rata-rata rating pengguna platform Xbox One dan PC adalah sama


```python
# Memfilter rating pengguna platform Xbox One dan PC

xone = new_data[new_data.platform == 'XOne']['user_score']
pc = new_data[new_data.platform == 'PC']['user_score']

```


```python
# Menghitung rata-rata dari xbox dan pc

xone_avg = new_data[new_data.platform == 'XOne']['user_score'].mean()
pc_avg = new_data[new_data.platform == 'PC']['user_score'].mean()
print('Rata-rata dari rating pengguna untuk platform XOne adalah {:.2f}'.format(xone_avg) + ' and ' + \
      'rata-rata dari rating pengguna untuk platform PC adalah {:.2f}'.format(pc_avg))
diff = (xone_avg - pc_avg) / xone_avg * 100
print('Persentasi Selisih Rata-rata Rating pengguna XOne dan PC adalah {:.2f}%'.format(diff))
```

    Rata-rata dari rating pengguna untuk platform XOne adalah 6.09 and rata-rata dari rating pengguna untuk platform PC adalah 5.68
    Persentasi Selisih Rata-rata Rating pengguna XOne dan PC adalah 6.73%


<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

Setelah menghitung rata-rata rating dari pengguna pada platform XOne dan PC, pada tahap selanjutnya kita ingin mengetahui apakah terdapat perbedaan yang signifikan dari nilai rata-rata pada platform XOne dan PC. alih-alih berasumsi berdasarkan rata-rata saja untuk memastikan, kita menggunakan data untuk melakukan uji statistik, pada percobaan ini dapat dirumuskan bahwa hipotesis null : tidak ada perbedaan antara rata-rata rating pengguna pada platform XOne dan PC dan Hipotesis alternatifnya adalah bahwa rata-rata rating pengguna platform XOne dan PC terdapat perbedaan. untuk melakukan pengujian hipotesis diatas kita menggunakan nilai signifikansi atau alpha 0.05 yang berarti dalam 5% tingkat kesalahannya. dimana kita akan menolak hipotesis nol ketika hipotesis alternatif nya benar lalu menggunakan uji-t untuk menguji hipotesis karena membandingkan rata-rata dua kelompok untuk menentukan apakah kedua kelompok ini berbeda satu sama lain.

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

1. Ho (Hipotesis nol)           = Rata-rata rating pengguna XOne = (sama dengan)  Rata-rata rating pengguna PC
_______________________________
2. H1 (Hipotesis alternatif)      = Rata-rata rating pengguna XOne <> (berbeda dengan) Rata-rata rating pengguna PC
_________________________________
3. α (alpha) tingkat signifikansi  = 0.05 
_________________________________
4. Jika p-value < (lebih kecil), maka hipotesis nol ditolak. jika p-value > (lebih besar) , maka hipotesis nol diterima.

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>


```python
# Menguji hipotesis

hipotesis_test(xone, pc, 0.05)
```

    Nilai p-value adalah : 0.016374465676443895
    Kita dapat menolak Ho


Kesimpulan :

Setelah menguji hipotesis null dengan menyatakan bahwa tidak ada perbedaan antara rata-rata rating pengguna platform XOne dan PC dengan menggunakan nilai signifikansi 0.05 dan hasilnya adalah p-value nya lebih kecil dari 0.05 maka kita dapat menolak Hipotesis null bahwa terdapat perbedaan yang signifikan secara statistik antara rata-rata rating pengguna platform XOne dan pc. 

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

### Rata-rata rating pengguna genre Action dan Sports berbeda.



```python
# Memfilter rating pengguna genre Action dan Sport 

action = new_data[new_data.genre == 'Action']['user_score']
sports = new_data[new_data.genre == 'Sports']['user_score']
```


```python
# Menghitung rata-rata pengguna genre Action dan Sport 

action_avg = new_data[new_data.genre == 'Action']['user_score'].mean()
sports_avg = new_data[new_data.genre == 'Sports']['user_score'].mean()
print('Rata-rata dari rating pengguna untuk genre Action adalah {:.2f}'.format(xone_avg) + ' and ' + \
      'rata-rata dari rating pengguna untuk genre Sport adalah {:.2f}'.format(pc_avg))
diff = (action_avg - sports_avg) / action_avg * 100
print('Persentasi Selisih Rata-rata Rating genre Action dan Sport adalah {:.2f}%'.format(diff))
```

    Rata-rata dari rating pengguna untuk genre Action adalah 6.09 and rata-rata dari rating pengguna untuk genre Sport adalah 5.68
    Persentasi Selisih Rata-rata Rating genre Action dan Sport adalah 13.16%


<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

Setelah menghitung rata-rata rating dari pengguna pada platform XOne dan PC, pada tahap selanjutnya kita ingin mengetahui apakah terdapat perbedaan yang signifikan dari nilai rata-rata pengguna genre action dan sport. alih-alih berasumsi berdasarkan rata-rata saja untuk memastikan, kita menggunakan data untuk melakukan uji statistik, pada percobaan ini dapat dirumuskan bahwa hipotesis null : Terdapat Perbedaan dari rata-rata pengguna genre action dan sport dan Hipotesis alternatifnya adalah bahwa rata-rata pengguna genre action dan sport tidak terdapat perbedaan. untuk melakukan pengujian hipotesis diatas kita menggunakan nilai signifikansi atau alpha 0.05 yang berarti dalam 5% tingkat kesalahannya. dimana kita akan menolak hipotesis nol ketika hipotesis alternatif nya benar lalu menggunakan uji-t untuk menguji hipotesis karena membandingkan rata-rata dua kelompok untuk menentukan apakah kedua kelompok ini berbeda satu sama lain.

1. Ho (Hipotesis nol)           = Rata-rata rating pengguna genre action <> (berbeda dengan) rata-rata rating pengguna genre sport
_______________________________
2. H1 (Hipotesis alternatif)      = Rata-rata rating pengguna genre action = (sama dengan) rata-rata rating  pengguna  genre sport
_________________________________
3. α (alpha) tingkat signifikansi  = 0.05 
_________________________________
4. Jika p-value < (lebih kecil), maka hipotesis nol ditolak. jika p-value > (lebih besar) , maka hipotesis nol diterima.

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>


```python
# Menguji hipotesis

hipotesis_test(action, sports, 0.05)
```

    Nilai p-value adalah : 4.3150499956308743e-08
    Kita dapat menolak Ho


Kesimpulan : 

Setelah menguji hipotesis null dengan menyatakan terdapat perbedaan antara rata-rata rating pengguna genre action dan sports dengan menggunakan nilai signifikansi 0.05 dan hasilnya adalah kita dapat menolak Hipotesis null bahwa terdapat perbedaan antara rata-rata rating pengguna genre action dan sport. 

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>

## Kesimpulan Umum

1. Tahap awal setelah mengimport library yang dibutuhkan lalu melihat informasi umum pada awal dataset didapati 6 kolom yang memiliki missing value serta terdapat tipe data yang kurang tepat sehingga perlu mengkonversi ke tipe data yang tepat, setelah mengkonversi column year_of_release dari float ke int dan column user_score ke str. seteleh mengidentifikasi lebih lanjut pada data yang hilang terdapat value yang hilang secara acak. lalu untuk penanganan nilai yang hilang kita mengisi nilai yang hilang secara acak dengan nilai acak berdasarkan nilai yang unik pada colum dimana fungsinya adalah jika nilai uniknya kosong maka akan diisi dengan median atau mode pada kolom yg teridentifikasi nilai yang hilang, lalu kita mendrop nilai yang hilan pada column nama dan genre dikarenakan nilai yang hilang kurang dari 1% serta singkatan 'tbd' di ubah ke nilai 'NaN' pada kumpulan data.
_______________________

2. Pada tahap analisis data, didapati game terbanyak dirilis pada tahun 2001 sd 2006. dan sebagian besar game pada chart lollipop dimana sebagian besar game dirilis antara tahun 2005 sd 2011. dan puncak game terbanayk dirilis pada tahun 2008, lalu dengan menganalisis variasi penjualan di seluruh platform, kita dapat melihat bahwa PS2, DS, PS3, Wii, dan X360 adalah lima platform teratas dalam hal total penjualan. Platform dengan penjualan paling sedikit adalah SCD, WS, 3DO, TG16, PCFX, dan GG. Kami juga menetapkan bahwa PC memiliki penjualan tertinggi di tahun 2011. PC merupakan platform dengan masa pakai terlama di antara penjualan platform lainnya selama sekitar 30 tahun. Biasanya membutuhkan waktu sekitar 6 tahun untuk platform baru muncul dan yang lama memudar.
________________________

3. Lalu pada tahap penjualan platform yang menguntungkan adalah PS4, PS3, XOne, 3DS, dan X360 berdasarkan total penjualan. TG16, 3DO, GG, dan PCFX adalah platform terburuk dalam hal total penjualan dengan nilai jauh di bawah rata-rata data. Kami melihat bagaimana ulasan pengguna dan profesional memengaruhi penjualan untuk satu platform populer untuk periode setelah tahun 2014. Kami menyimpulkan setelah menghitung p-value dan analisis statistik bahwa ada hubungan linier yang signifikan antara ulasan pengguna dan profesional dan Total penjualan untuk produk. Karenanya, ulasan pengguna memengaruhi total penjualan.
_________________________

4. Dengan memerisa data lebih lanjut dan mengamati variasi dari pangsa pasar di lima platform teratas dari satu wilayah ke wilayah lain. Di wilayah NA PS4 memiliki pangsa pasar terbanyak. Di wilayah UE, PS4 memiliki pangsa pasar terbanyak. Di wilayah JP, 3DS memiliki pangsa pasar terbanyak. Wilayah NA dan wilayah UE sangat mirip, lalu pada wilayah JP berbeda dengan region lain karena memiliki genre Fighting yang tidak ada di wilayah EU dan NA, dan genre Lain-lain yang tidak ada di region EU. Melihat peringkat wilayah, kami menemukan bahwa peringkat ESRB (Entertainment Software Rating Board ) mempengaruhi penjualan di masing-masing wilayah. Di masing-masing dari ketiga wilayah tersebut, rating E, T, M, dan E10+ mendapatkan penjualan tertinggi.
________________________

5. Setelah menguji hipotesis null untuk menentukan rata-rata peringkat pengguna platform XOne  dan PC dengan menyatakan bahwa tidak ada perbedaan antara rata-rata rating pengguna platform XOne dan PC dengan menggunakan nilai signifikansi 0.05 dan hasilnya adalah p-value nya lebih kecil dari 0.05 maka kita dapat menolak Hipotesis null bahwa terdapat perbedaan yang signifikan secara statistik antara rata-rata rating pengguna platform XOne dan pc. dan Setelah menguji hipotesis null untuk menentukan rata-rata peringkat untuk genre  untuk genre Action dan sport berbeda dengan menyatakan terdapat perbedaan antara rata-rata rating pengguna genre action dan sports dengan menggunakan nilai signifikansi 0.05 dan hasilnya adalah kita dapat menolak Hipotesis null bahwa terdapat perbedaan antara rata-rata rating pengguna genre action dan sport.

<div class="alert alert-success">
<b>Chamdani's comment v.1</b> <a class="tocSkip"></a>

Kerja bagus!

</div>


```python

```
