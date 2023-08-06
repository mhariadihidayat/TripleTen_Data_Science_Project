# Ekplorasi Faktor Penjualan Mobil Terhadap Harga pada Usia, Jarak tempuh, Kondisi, Tipe Transmisi, dan Warnanya.

1. Pada tahap pertama Pra-pemprosesan data:
     a. Memuat Library.
     b. Eksplorasi Data.
     c. Kesimpulan Awal Untuk proses selanjutnya.



2. Pada tahap Explorasi Data / Data Quality Checking
    a. Mengatasi nilai yang hilang
    b. Memperbaiki tipe data
    c. Memperbaiki kualitas data
    d. Memeriksa data yang sudah bersih.



3. Pada tahap Data Cleansing:
    a. Mempelajari Parameter
    b. Menangani outlier



4. Explorasi data terkait dengan faktor penjualan mobil terhadap harga pada Usia, jarak tempuh, kondisi, tipe transmisi dan warnanya. 

## Pra-pemrosesan

Memuat Perpustakan 
1. pandas
2. numpy
3. matplotlin.pyplot
4. datetime


```python
# Muat semua library

# import pandas and numpy untuk proses dan manipulasi data
import pandas as pd
import numpy as np

# import scipy untuk perhitungan statistika zscore
from scipy import stats

# import matplotlib untuk data visualisasi
import matplotlib.pyplot as plt 
%matplotlib inline

# Import seaborn for statistika data visualisasi
import seaborn as sns

# import date dan time untuk merubah tipe data
import time
import datetime
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
```

### Memuat Data

Memuat Data dari pandas menjadi DataFrame


```python
# Muat file data menjadi DataFrame
try: 
  data = pd.read_csv('/datasets/vehicles_us.csv')

except:
  data = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/My Project Practicum/Project Vehicles by Practicum Not valid/vehicles_us.csv')
 
  
```

### Mengeksplorasi Data Awal

*Dataset* Anda berisi kolom-kolom berikut: 


- `price`
- `model_year`
- `model`
- `condition`
- `cylinders`
- `fuel` — gas, disel, dan lain-lain.
- `odometer` — jarak tempuh kendaraan saat iklan ditayangkan 
- `transmission`
- `paint_color`
- `is_4wd` — apakah kendaraan memiliki penggerak 4 roda (tipe Boolean)
- `date_posted` — tanggal iklan ditayangkan
- `days_listed` — jumlah hari iklan ditayangkan hingga dihapus



```python
# tampilkan informasi/rangkuman umum tentang DataFrame

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51525 entries, 0 to 51524
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   price         51525 non-null  int64  
     1   model_year    47906 non-null  float64
     2   model         51525 non-null  object 
     3   condition     51525 non-null  object 
     4   cylinders     46265 non-null  float64
     5   fuel          51525 non-null  object 
     6   odometer      43633 non-null  float64
     7   transmission  51525 non-null  object 
     8   type          51525 non-null  object 
     9   paint_color   42258 non-null  object 
     10  is_4wd        25572 non-null  float64
     11  date_posted   51525 non-null  object 
     12  days_listed   51525 non-null  int64  
    dtypes: float64(4), int64(2), object(7)
    memory usage: 5.1+ MB



```python
# tampilkan sampel data

data.sample(frac=2, replace=True, random_state=1)
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33003</th>
      <td>4295</td>
      <td>2004.0</td>
      <td>chevrolet suburban</td>
      <td>good</td>
      <td>8.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2019-03-13</td>
      <td>22</td>
    </tr>
    <tr>
      <th>12172</th>
      <td>1</td>
      <td>2016.0</td>
      <td>ram 3500</td>
      <td>excellent</td>
      <td>10.0</td>
      <td>gas</td>
      <td>43200.0</td>
      <td>other</td>
      <td>truck</td>
      <td>white</td>
      <td>1.0</td>
      <td>2018-05-13</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5192</th>
      <td>2999</td>
      <td>2004.0</td>
      <td>toyota sienna</td>
      <td>like new</td>
      <td>6.0</td>
      <td>gas</td>
      <td>185696.0</td>
      <td>automatic</td>
      <td>mini-van</td>
      <td>grey</td>
      <td>NaN</td>
      <td>2019-02-26</td>
      <td>20</td>
    </tr>
    <tr>
      <th>32511</th>
      <td>15995</td>
      <td>2016.0</td>
      <td>chevrolet camaro lt coupe 2d</td>
      <td>like new</td>
      <td>6.0</td>
      <td>gas</td>
      <td>30000.0</td>
      <td>manual</td>
      <td>coupe</td>
      <td>black</td>
      <td>NaN</td>
      <td>2018-07-09</td>
      <td>44</td>
    </tr>
    <tr>
      <th>50057</th>
      <td>5900</td>
      <td>2005.0</td>
      <td>toyota highlander</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>195643.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>silver</td>
      <td>1.0</td>
      <td>2018-05-16</td>
      <td>26</td>
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
    </tr>
    <tr>
      <th>44639</th>
      <td>3900</td>
      <td>NaN</td>
      <td>ford f250 super duty</td>
      <td>fair</td>
      <td>8.0</td>
      <td>gas</td>
      <td>82300.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>white</td>
      <td>1.0</td>
      <td>2018-10-26</td>
      <td>94</td>
    </tr>
    <tr>
      <th>28674</th>
      <td>15500</td>
      <td>2007.0</td>
      <td>toyota tundra</td>
      <td>excellent</td>
      <td>8.0</td>
      <td>gas</td>
      <td>162000.0</td>
      <td>automatic</td>
      <td>offroad</td>
      <td>white</td>
      <td>1.0</td>
      <td>2018-11-02</td>
      <td>58</td>
    </tr>
    <tr>
      <th>36544</th>
      <td>7995</td>
      <td>2005.0</td>
      <td>chrysler 300</td>
      <td>good</td>
      <td>8.0</td>
      <td>gas</td>
      <td>183080.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>grey</td>
      <td>NaN</td>
      <td>2018-12-24</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4949</th>
      <td>16395</td>
      <td>2017.0</td>
      <td>ford escape</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-06-21</td>
      <td>109</td>
    </tr>
    <tr>
      <th>10229</th>
      <td>2600</td>
      <td>1998.0</td>
      <td>toyota 4runner</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>191000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>red</td>
      <td>NaN</td>
      <td>2019-04-01</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
<p>103050 rows × 13 columns</p>
</div>



Kesimpulan awal :

Dari hasil sampel data terdapat nilai yang hilang, sehingga mungkin memerlukan identifikasi pada data lebih lanjut.



```python
# Memeriksa column yang tidak sesuai dengan tipe datanya

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51525 entries, 0 to 51524
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   price         51525 non-null  int64  
     1   model_year    47906 non-null  float64
     2   model         51525 non-null  object 
     3   condition     51525 non-null  object 
     4   cylinders     46265 non-null  float64
     5   fuel          51525 non-null  object 
     6   odometer      43633 non-null  float64
     7   transmission  51525 non-null  object 
     8   type          51525 non-null  object 
     9   paint_color   42258 non-null  object 
     10  is_4wd        25572 non-null  float64
     11  date_posted   51525 non-null  object 
     12  days_listed   51525 non-null  int64  
    dtypes: float64(4), int64(2), object(7)
    memory usage: 5.1+ MB


Kesimpulan awal :

1. Terdapat tipe data yang tidak sesuai yaitu date_posted dengan tipe data object, yang seharusnya tipe data timestamp.


```python
# Menampilkan Column dengan jumlah nilai yang hilang

data.isna().sum()
```




    price               0
    model_year       3619
    model               0
    condition           0
    cylinders        5260
    fuel                0
    odometer         7892
    transmission        0
    type                0
    paint_color      9267
    is_4wd          25953
    date_posted         0
    days_listed         0
    dtype: int64




```python
# Memeriksa jumlah panjang baris column pada Dataset

data.shape
```




    (51525, 13)



Kesimpulan awal:

Terdapat nilai yang hilang di beberapa column dengan jumlah nilai column yang hilang terbesar  25953

Memfilter dataset berdasarkan column yang teridentifikasi nilai yang hilang apakah terdapat pola atau hilang secara acak. 


```python
# Filter Data yang hilang berdasarkan column yang nilainya teridentifikasi terdapat nilai yang hilang

data.loc[(data['model_year'].isna()) & (data['cylinders'].isna())]

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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>72</th>
      <td>3650</td>
      <td>NaN</td>
      <td>subaru impreza</td>
      <td>excellent</td>
      <td>NaN</td>
      <td>gas</td>
      <td>74000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>blue</td>
      <td>1.0</td>
      <td>2018-08-07</td>
      <td>60</td>
    </tr>
    <tr>
      <th>159</th>
      <td>23300</td>
      <td>NaN</td>
      <td>nissan frontier crew cab sv</td>
      <td>good</td>
      <td>NaN</td>
      <td>gas</td>
      <td>NaN</td>
      <td>other</td>
      <td>pickup</td>
      <td>grey</td>
      <td>1.0</td>
      <td>2018-07-24</td>
      <td>73</td>
    </tr>
    <tr>
      <th>370</th>
      <td>4700</td>
      <td>NaN</td>
      <td>kia soul</td>
      <td>good</td>
      <td>NaN</td>
      <td>gas</td>
      <td>NaN</td>
      <td>manual</td>
      <td>sedan</td>
      <td>white</td>
      <td>NaN</td>
      <td>2019-01-14</td>
      <td>50</td>
    </tr>
    <tr>
      <th>418</th>
      <td>4998</td>
      <td>NaN</td>
      <td>toyota corolla</td>
      <td>good</td>
      <td>NaN</td>
      <td>gas</td>
      <td>44442.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>grey</td>
      <td>NaN</td>
      <td>2019-04-19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>664</th>
      <td>5000</td>
      <td>NaN</td>
      <td>toyota highlander</td>
      <td>excellent</td>
      <td>NaN</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>blue</td>
      <td>NaN</td>
      <td>2018-06-27</td>
      <td>14</td>
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
    </tr>
    <tr>
      <th>50643</th>
      <td>21499</td>
      <td>NaN</td>
      <td>ram 2500</td>
      <td>good</td>
      <td>NaN</td>
      <td>diesel</td>
      <td>165831.0</td>
      <td>manual</td>
      <td>truck</td>
      <td>red</td>
      <td>1.0</td>
      <td>2018-08-14</td>
      <td>91</td>
    </tr>
    <tr>
      <th>50836</th>
      <td>39488</td>
      <td>NaN</td>
      <td>ford f350</td>
      <td>like new</td>
      <td>NaN</td>
      <td>gas</td>
      <td>32000.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-06-23</td>
      <td>6</td>
    </tr>
    <tr>
      <th>50924</th>
      <td>1999</td>
      <td>NaN</td>
      <td>jeep grand cherokee laredo</td>
      <td>good</td>
      <td>NaN</td>
      <td>gas</td>
      <td>199312.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>blue</td>
      <td>1.0</td>
      <td>2019-01-05</td>
      <td>69</td>
    </tr>
    <tr>
      <th>51125</th>
      <td>20900</td>
      <td>NaN</td>
      <td>ford f-150</td>
      <td>excellent</td>
      <td>NaN</td>
      <td>gas</td>
      <td>53209.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-08-04</td>
      <td>87</td>
    </tr>
    <tr>
      <th>51351</th>
      <td>23900</td>
      <td>NaN</td>
      <td>chevrolet silverado 2500hd</td>
      <td>good</td>
      <td>NaN</td>
      <td>diesel</td>
      <td>166053.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>silver</td>
      <td>1.0</td>
      <td>2018-08-31</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
<p>363 rows × 13 columns</p>
</div>




```python
# Filter Data hilang berdasarkan column odometer yang teridentifikasi terdapat nilai yang hilang

data.loc[(data['model_year'].isna()) & (data['odometer'].isna())]

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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>23300</td>
      <td>NaN</td>
      <td>nissan frontier crew cab sv</td>
      <td>good</td>
      <td>NaN</td>
      <td>gas</td>
      <td>NaN</td>
      <td>other</td>
      <td>pickup</td>
      <td>grey</td>
      <td>1.0</td>
      <td>2018-07-24</td>
      <td>73</td>
    </tr>
    <tr>
      <th>260</th>
      <td>14975</td>
      <td>NaN</td>
      <td>toyota 4runner</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2018-05-13</td>
      <td>57</td>
    </tr>
    <tr>
      <th>370</th>
      <td>4700</td>
      <td>NaN</td>
      <td>kia soul</td>
      <td>good</td>
      <td>NaN</td>
      <td>gas</td>
      <td>NaN</td>
      <td>manual</td>
      <td>sedan</td>
      <td>white</td>
      <td>NaN</td>
      <td>2019-01-14</td>
      <td>50</td>
    </tr>
    <tr>
      <th>586</th>
      <td>26000</td>
      <td>NaN</td>
      <td>toyota rav4</td>
      <td>like new</td>
      <td>4.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-08-09</td>
      <td>29</td>
    </tr>
    <tr>
      <th>659</th>
      <td>8400</td>
      <td>NaN</td>
      <td>volkswagen jetta</td>
      <td>good</td>
      <td>4.0</td>
      <td>diesel</td>
      <td>NaN</td>
      <td>manual</td>
      <td>wagon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-10-22</td>
      <td>37</td>
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
    </tr>
    <tr>
      <th>51195</th>
      <td>21999</td>
      <td>NaN</td>
      <td>ram 2500</td>
      <td>good</td>
      <td>6.0</td>
      <td>diesel</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>truck</td>
      <td>white</td>
      <td>1.0</td>
      <td>2018-05-10</td>
      <td>35</td>
    </tr>
    <tr>
      <th>51222</th>
      <td>1000</td>
      <td>NaN</td>
      <td>acura tl</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>grey</td>
      <td>NaN</td>
      <td>2018-12-09</td>
      <td>23</td>
    </tr>
    <tr>
      <th>51257</th>
      <td>6500</td>
      <td>NaN</td>
      <td>toyota corolla</td>
      <td>good</td>
      <td>4.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>white</td>
      <td>NaN</td>
      <td>2018-10-16</td>
      <td>75</td>
    </tr>
    <tr>
      <th>51295</th>
      <td>3850</td>
      <td>NaN</td>
      <td>hyundai elantra</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2019-03-16</td>
      <td>83</td>
    </tr>
    <tr>
      <th>51399</th>
      <td>4400</td>
      <td>NaN</td>
      <td>kia sorento</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2018-08-21</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>549 rows × 13 columns</p>
</div>




```python
# Filter Data hilang berdasarkan column paint color yang teridentifikasi terdapat nilai yang hilang

data.loc[(data['model_year'].isna()) & (data['paint_color'].isna())]
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>116</th>
      <td>25300</td>
      <td>NaN</td>
      <td>chevrolet camaro lt coupe 2d</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>3568.0</td>
      <td>other</td>
      <td>coupe</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-16</td>
      <td>34</td>
    </tr>
    <tr>
      <th>165</th>
      <td>22000</td>
      <td>NaN</td>
      <td>ford f350 super duty</td>
      <td>good</td>
      <td>8.0</td>
      <td>diesel</td>
      <td>163000.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2019-02-05</td>
      <td>38</td>
    </tr>
    <tr>
      <th>397</th>
      <td>14995</td>
      <td>NaN</td>
      <td>chevrolet camaro</td>
      <td>excellent</td>
      <td>8.0</td>
      <td>gas</td>
      <td>95000.0</td>
      <td>automatic</td>
      <td>coupe</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-29</td>
      <td>15</td>
    </tr>
    <tr>
      <th>443</th>
      <td>2025</td>
      <td>NaN</td>
      <td>chevrolet tahoe</td>
      <td>good</td>
      <td>8.0</td>
      <td>gas</td>
      <td>151000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-10-24</td>
      <td>31</td>
    </tr>
    <tr>
      <th>586</th>
      <td>26000</td>
      <td>NaN</td>
      <td>toyota rav4</td>
      <td>like new</td>
      <td>4.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-08-09</td>
      <td>29</td>
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
    </tr>
    <tr>
      <th>51312</th>
      <td>1800</td>
      <td>NaN</td>
      <td>hyundai santa fe</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>287000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-13</td>
      <td>100</td>
    </tr>
    <tr>
      <th>51339</th>
      <td>19890</td>
      <td>NaN</td>
      <td>toyota tundra</td>
      <td>excellent</td>
      <td>8.0</td>
      <td>gas</td>
      <td>127405.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2019-02-02</td>
      <td>69</td>
    </tr>
    <tr>
      <th>51385</th>
      <td>3495</td>
      <td>NaN</td>
      <td>jeep liberty</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>129644.0</td>
      <td>automatic</td>
      <td>wagon</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-08-22</td>
      <td>28</td>
    </tr>
    <tr>
      <th>51396</th>
      <td>14995</td>
      <td>NaN</td>
      <td>ford f-150</td>
      <td>good</td>
      <td>8.0</td>
      <td>gas</td>
      <td>123676.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-07-12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>51438</th>
      <td>6500</td>
      <td>NaN</td>
      <td>chevrolet silverado</td>
      <td>fair</td>
      <td>8.0</td>
      <td>gas</td>
      <td>187900.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-05-05</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
<p>652 rows × 13 columns</p>
</div>




```python
# Filter Data hilang berdasarkan column is_4wd yang teridentifikasi terdapat nilai yang hilang

data.loc[(data['model_year'].isna()) & (data['is_4wd'].isna())]
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>12800</td>
      <td>NaN</td>
      <td>ford f-150</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>108500.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>white</td>
      <td>NaN</td>
      <td>2018-09-23</td>
      <td>15</td>
    </tr>
    <tr>
      <th>84</th>
      <td>4995</td>
      <td>NaN</td>
      <td>hyundai elantra</td>
      <td>like new</td>
      <td>4.0</td>
      <td>gas</td>
      <td>151223.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>custom</td>
      <td>NaN</td>
      <td>2018-09-15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>116</th>
      <td>25300</td>
      <td>NaN</td>
      <td>chevrolet camaro lt coupe 2d</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>3568.0</td>
      <td>other</td>
      <td>coupe</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-16</td>
      <td>34</td>
    </tr>
    <tr>
      <th>164</th>
      <td>2500</td>
      <td>NaN</td>
      <td>toyota camry</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>150000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2018-06-13</td>
      <td>68</td>
    </tr>
    <tr>
      <th>186</th>
      <td>3000</td>
      <td>NaN</td>
      <td>honda accord</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>204000.0</td>
      <td>automatic</td>
      <td>coupe</td>
      <td>blue</td>
      <td>NaN</td>
      <td>2018-05-18</td>
      <td>21</td>
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
    </tr>
    <tr>
      <th>51357</th>
      <td>7995</td>
      <td>NaN</td>
      <td>toyota prius</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>hybrid</td>
      <td>106250.0</td>
      <td>automatic</td>
      <td>hatchback</td>
      <td>red</td>
      <td>NaN</td>
      <td>2019-01-28</td>
      <td>24</td>
    </tr>
    <tr>
      <th>51378</th>
      <td>5495</td>
      <td>NaN</td>
      <td>honda civic</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>gas</td>
      <td>130000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>brown</td>
      <td>NaN</td>
      <td>2018-06-17</td>
      <td>47</td>
    </tr>
    <tr>
      <th>51399</th>
      <td>4400</td>
      <td>NaN</td>
      <td>kia sorento</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2018-08-21</td>
      <td>23</td>
    </tr>
    <tr>
      <th>51411</th>
      <td>7995</td>
      <td>NaN</td>
      <td>ford taurus</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>149462.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2018-09-16</td>
      <td>86</td>
    </tr>
    <tr>
      <th>51508</th>
      <td>4950</td>
      <td>NaN</td>
      <td>chrysler town &amp; country</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>150000.0</td>
      <td>automatic</td>
      <td>mini-van</td>
      <td>silver</td>
      <td>NaN</td>
      <td>2018-06-30</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
<p>1811 rows × 13 columns</p>
</div>



### Kesimpulan dan Langkah-Langkah Selanjutnya

Kesimpulan mengenai data awal :

1. Terdapat nilai yang hilang secara acak (MAR) pada column model_year, cylinders, odometer, paint_color, is_4wd.

2. Terdapat tipe data float yang harus harus di konversi ke tipe data int

3. Jika nilai yang hilang berada pada tipe data kategorik dan nilai yang hilang teridentifikasi secara acak, maka penanganan pada data dapat diganti dengan nilai default seperti string kosong atau teks tertentu dengan pengindeksan boolean serta mengisi dengan nilai yang paling banyak muncul.

4. Jika nilai yang hilang berada pada tipe data kuantitatif maka penanganan pada data dapat menggunakan nilai representatif mean atau median untuk mengisi value yang hilang tersebut. lalu jika value tidak memiliki outlier nilai yang hilang dapat diganti dengan mean/ rata-rata, jika value memiliki nilai outlier teridentifikasi signifikan maka nilai yang hilang dapat diganti dengan median. 

5. Kemungkinan Faktor penyebab nilai yang hilang pada column yang teridentifikasi :

    a. Pengguna lupa mengisi column.

    b. Data hilang saat mentransfer secara manual dari database.

    c. Terjadi kesalahan pemrograman.





## Mengatasi Nilai-Nilai yang Hilang (Jika Ada)

Mengidentifikasi nilai yang hilang 


```python
# Memeriksa nilai yang hilang pada dataset

data.isnull().sum()
```




    price               0
    model_year       3619
    model               0
    condition           0
    cylinders        5260
    fuel                0
    odometer         7892
    transmission        0
    type                0
    paint_color      9267
    is_4wd          25953
    date_posted         0
    days_listed         0
    dtype: int64




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
# Menampilkan hasil dari fungsi

missing_values_table(data)
```

    Dataset yang dipilih 13  column.
    Terdapat 5 column yang memiliki nilai yang hilang.





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
      <th>is_4wd</th>
      <td>25953</td>
      <td>50.4</td>
    </tr>
    <tr>
      <th>paint_color</th>
      <td>9267</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>odometer</th>
      <td>7892</td>
      <td>15.3</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>5260</td>
      <td>10.2</td>
    </tr>
    <tr>
      <th>model_year</th>
      <td>3619</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Memeriksa Distribusi statistik dataset
data.describe()
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
      <th>price</th>
      <th>model_year</th>
      <th>cylinders</th>
      <th>odometer</th>
      <th>is_4wd</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>51525.000000</td>
      <td>47906.000000</td>
      <td>46265.000000</td>
      <td>43633.000000</td>
      <td>25572.0</td>
      <td>51525.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12132.464920</td>
      <td>2009.750470</td>
      <td>6.125235</td>
      <td>115553.461738</td>
      <td>1.0</td>
      <td>39.55476</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10040.803015</td>
      <td>6.282065</td>
      <td>1.660360</td>
      <td>65094.611341</td>
      <td>0.0</td>
      <td>28.20427</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1908.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5000.000000</td>
      <td>2006.000000</td>
      <td>4.000000</td>
      <td>70000.000000</td>
      <td>1.0</td>
      <td>19.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9000.000000</td>
      <td>2011.000000</td>
      <td>6.000000</td>
      <td>113000.000000</td>
      <td>1.0</td>
      <td>33.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16839.000000</td>
      <td>2014.000000</td>
      <td>8.000000</td>
      <td>155000.000000</td>
      <td>1.0</td>
      <td>53.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>375000.000000</td>
      <td>2019.000000</td>
      <td>12.000000</td>
      <td>990000.000000</td>
      <td>1.0</td>
      <td>271.00000</td>
    </tr>
  </tbody>
</table>
</div>



1.  Mengatasi nilai-nilai yang hilang pada column is_4wd




```python
# Memeriksa nilai yang unik pada column is_4wd

data['is_4wd'].sort_values().unique()
```




    array([ 1., nan])




```python
# Mempelajari nilai yang hilang pada column s_4wd

data['is_4wd'].value_counts(dropna=False)
```




    NaN    25953
    1.0    25572
    Name: is_4wd, dtype: int64




```python
# Melihat distribusi nilai pada column is_4wd

data['is_4wd'].describe()
```




    count    25572.0
    mean         1.0
    std          0.0
    min          1.0
    25%          1.0
    50%          1.0
    75%          1.0
    max          1.0
    Name: is_4wd, dtype: float64



Kesimpulan: 

Setelah memeriksa nilai yang hilang kita pada column is_4wd  dengan tipe data kategoris, dimana terdapat 50% nilai yang hilang dimana nilai tersebut merepresentasikan bahwa kendaaraan yang tidak merupakan 4wd, dimana True adalah 1 dan False adalah 0. maka dari itu kita akan mengisi nilainya dengan false atau 0


```python
# Mengisi nilai yang hilang pad column is_4wd 

data['is_4wd'].fillna(0, inplace = True)
```


```python
# Memeriksa nilai yang hilang setelah di isi dengan value 0

data.head()
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9400</td>
      <td>2011.0</td>
      <td>bmw x5</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>145000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2018-06-23</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25500</td>
      <td>NaN</td>
      <td>ford f-150</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>88705.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>white</td>
      <td>1.0</td>
      <td>2018-10-19</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5500</td>
      <td>2013.0</td>
      <td>hyundai sonata</td>
      <td>like new</td>
      <td>4.0</td>
      <td>gas</td>
      <td>110000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>red</td>
      <td>0.0</td>
      <td>2019-02-07</td>
      <td>79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1500</td>
      <td>2003.0</td>
      <td>ford f-150</td>
      <td>fair</td>
      <td>8.0</td>
      <td>gas</td>
      <td>NaN</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2019-03-22</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14900</td>
      <td>2017.0</td>
      <td>chrysler 200</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>gas</td>
      <td>80903.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>0.0</td>
      <td>2019-04-02</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



2. Mengidentifikasi Nilai yang hilang dengan mengidentifikasi column lain untuk menjadi referensi 


```python
# Mengidentifikasi price mobil yang tertinggi 

data.loc[data['price'] == 375000]
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12504</th>
      <td>375000</td>
      <td>1999.0</td>
      <td>nissan frontier</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>115000.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>blue</td>
      <td>1.0</td>
      <td>2018-05-19</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mengidentifikasi Jarak tempuh mobil yang tertinggi 

data.loc[data['odometer']== 990000]
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17869</th>
      <td>59900</td>
      <td>1964.0</td>
      <td>chevrolet corvette</td>
      <td>like new</td>
      <td>NaN</td>
      <td>gas</td>
      <td>990000.0</td>
      <td>automatic</td>
      <td>convertible</td>
      <td>red</td>
      <td>0.0</td>
      <td>2018-06-17</td>
      <td>28</td>
    </tr>
    <tr>
      <th>40729</th>
      <td>4700</td>
      <td>2013.0</td>
      <td>chevrolet cruze</td>
      <td>good</td>
      <td>6.0</td>
      <td>gas</td>
      <td>990000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>0.0</td>
      <td>2018-05-02</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



a. Dengan menggunakan metode describe, kita dapat mengidentifikasi kesenjangan data, contohnya pada nilai maksimal dari column Price (harga mobil) yakni 375000, dengan memfilter harga tertinggi pada mobil nissan frontier dengan model tahun 1999 dengan jarak tempuh 115000 dimana dapat dilihat pada tabel distribusi statistik dataset nilai maksimal dari column odometer adalah 990000, ini menunjukkan bahwa terdapat nilai yang outlier pada column price, oleh karena itu nilai yang hilang pada column model_year dan cylinders tidak bisa diisi dengan nilai rata-rata / mean sehingga alternatif lain dalam menangani nilai yang hilang kita dapat mengisi nilai median dengan metode conditional imputation. 


b. Untuk nilai yang hilang pada column paint_color dikarenakan tipe datanya kategoris kita tidak dapat mengisinya dengan median, mean pada tipe data categorical, sehingga kita dapat mengisinya dengan kategori baru. 


c. untuk nilai yang hilang pada column odometer dikarenakan terdapat nilai yang outlier, maka kita dapat mengisinya dengan median atau nilai tengah pada distribusi statistik pada column tersebut dengan metode conditional imputation. 



2. Mengatasi nilai yang hilang pada column model_year dengan metode conditional imputation yakni dengan column condition sebagai referensi


```python
# Melihat value pada column condition

sorted(data['condition'].unique())

```




    ['excellent', 'fair', 'good', 'like new', 'new', 'salvage']




```python
# Membuat Variable manipulasi unutk menjalankan metode conditional imputation

data_drop = data.dropna()
data_drop.head()
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>5500</td>
      <td>2013.0</td>
      <td>hyundai sonata</td>
      <td>like new</td>
      <td>4.0</td>
      <td>gas</td>
      <td>110000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>red</td>
      <td>0.0</td>
      <td>2019-02-07</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14900</td>
      <td>2017.0</td>
      <td>chrysler 200</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>gas</td>
      <td>80903.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>0.0</td>
      <td>2019-04-02</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14990</td>
      <td>2014.0</td>
      <td>chrysler 300</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>57954.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>1.0</td>
      <td>2018-06-20</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12990</td>
      <td>2015.0</td>
      <td>toyota camry</td>
      <td>excellent</td>
      <td>4.0</td>
      <td>gas</td>
      <td>79212.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>white</td>
      <td>0.0</td>
      <td>2018-12-27</td>
      <td>73</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15990</td>
      <td>2013.0</td>
      <td>honda pilot</td>
      <td>excellent</td>
      <td>6.0</td>
      <td>gas</td>
      <td>109473.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>black</td>
      <td>1.0</td>
      <td>2019-01-07</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menghitung nilai median dari column model_year berdasarkan column condition

print('nilai median dari model_year adalah:')
data_drop.pivot_table(index='condition', values='model_year', aggfunc='median')
```

    nilai median dari model_year adalah:





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
      <th>model_year</th>
    </tr>
    <tr>
      <th>condition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>fair</th>
      <td>2003.0</td>
    </tr>
    <tr>
      <th>good</th>
      <td>2009.0</td>
    </tr>
    <tr>
      <th>like new</th>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>new</th>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>salvage</th>
      <td>2006.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menerapkan nilai median pada column model_year berdasarkan faktor yang telah diidentifikasi

data['model_year'] = data.groupby('condition')['model_year'].transform(lambda x: x.fillna(x.median()))
```


```python
# Menampilkan column model_year setelah menerapkan metode conditional imputation

data['model_year'].isna().sum()
```




    0



3. Mengatasi nilai yang hilang pada column odometer dengan metode conditional imputation yakni dengan column condition sebagai referensi


```python
# Menghitung nilai median dari column odometer berdasarkan column condition

print('nilai median dari odometer adalah:')
data_drop.pivot_table(index='condition', values='odometer', aggfunc='median')
```

    nilai median dari odometer adalah:





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
      <th>odometer</th>
    </tr>
    <tr>
      <th>condition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>excellent</th>
      <td>104174.0</td>
    </tr>
    <tr>
      <th>fair</th>
      <td>180000.0</td>
    </tr>
    <tr>
      <th>good</th>
      <td>129000.0</td>
    </tr>
    <tr>
      <th>like new</th>
      <td>72000.0</td>
    </tr>
    <tr>
      <th>new</th>
      <td>2600.0</td>
    </tr>
    <tr>
      <th>salvage</th>
      <td>148000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menerapkan nilai median pada column odometer berdasarkan faktor yang telah diidentifikasi

data['odometer'] = data.groupby('condition')['odometer'].transform(lambda x: x.fillna(x.median()))
```


```python
# Menampilkan column odometer setelah menerapkan metode conditional imputation

data['odometer'].isna().sum()
```




    0



4. Mengatasi nilai yang hilang pada column paint_color


```python
# Mengisi nilai yang hilang pada column paint_color dengan string 'others'

data['paint_color'] = data['paint_color'].fillna(value='others')
```

5. Mengatasi nilai yang hilang pada column cylinders


```python
# Melihat value pada column cylinders

sorted(data['cylinders'].unique())
```




    [3.0, 4.0, 5.0, 6.0, 8.0, nan, 10.0, 12.0]




```python
# Menghitung nilai median dari column cylinders berdasarkan column model

print('nilai median dari cylinders adalah:')
data_drop.pivot_table(index='model', values='cylinders', aggfunc='median')
```

    nilai median dari cylinders adalah:





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
      <th>cylinders</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acura tl</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>bmw x5</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>buick enclave</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>cadillac escalade</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>chevrolet camaro</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>toyota sienna</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>toyota tacoma</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>toyota tundra</th>
      <td>8.0</td>
    </tr>
    <tr>
      <th>volkswagen jetta</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>volkswagen passat</th>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>99 rows × 1 columns</p>
</div>




```python
# Menerapkan nilai median pada column odometer berdasarkan faktor yang telah diidentifikasi

data['cylinders'] = data.groupby('model')['cylinders'].transform(lambda x: x.fillna(x.median()))
```


```python
# Menampilkan column odometer setelah menerapkan metode conditional imputation

data['cylinders'].isna().sum()
```




    0



Menampilkan distribusi nilai statistik pada dataset setalah manipulasi data 


```python
# Melihat distribusi statistik pada dataset setelah manipulasi data :

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51525 entries, 0 to 51524
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   price         51525 non-null  int64  
     1   model_year    51525 non-null  float64
     2   model         51525 non-null  object 
     3   condition     51525 non-null  object 
     4   cylinders     51525 non-null  float64
     5   fuel          51525 non-null  object 
     6   odometer      51525 non-null  float64
     7   transmission  51525 non-null  object 
     8   type          51525 non-null  object 
     9   paint_color   51525 non-null  object 
     10  is_4wd        51525 non-null  float64
     11  date_posted   51525 non-null  object 
     12  days_listed   51525 non-null  int64  
    dtypes: float64(4), int64(2), object(7)
    memory usage: 5.1+ MB


Kesimpulan :

setelah kita mengidentifikasi nilai yang hilang pada dataset, kita menggunakan beberapa metode untuk menangani nilai yang hilang berdasarkan kasus perkasus. contohnya kita memanipulasi data dengan metode imputation conditional pada column model_year, cylinders dan mengganti nilai yang hilang menggunakan nilai mediannya dan membuat kategori baru pada column paint_color dengan nama 'others'.



## Memperbaiki Tipe Data


```python
# Memeriksa Tipe Data pada dataset

data.dtypes
```




    price             int64
    model_year      float64
    model            object
    condition        object
    cylinders       float64
    fuel             object
    odometer        float64
    transmission     object
    type             object
    paint_color      object
    is_4wd          float64
    date_posted      object
    days_listed       int64
    dtype: object



Kesimpulan :

Pada poin ini kita memerlukan perubahan pada tipe data yakni dari float64 menjadi int64, column yang tipe datanya perlu dirubah adalah model_year, cylinders, odometer dan is_4wd. alasan perubahan tipe data ini dikarenakan pada saat melakukan aritmatika lebih baik menggunakan bilangan bulat dari pada pecahan atau desimal.


## Memperbaiki kualitas data

Menambahkan beberapa faktor terkait data agar memudahkan dalam menganalisis.


```python
# Merubah column model_year, cylinders, odometer dan is_4wd ke tipe data int64

data['price'] = data['price'].astype('float64')
data['model_year'] = data['model_year'].astype('int64')
data['cylinders'] = data['cylinders'].astype('int64')
data['odometer'] = data['odometer'].astype('float64')
data['is_4wd'] = data['is_4wd'].astype('bool')
```


```python
# Menampilkan dataset yang telah dirubah tipenya

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51525 entries, 0 to 51524
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   price         51525 non-null  float64
     1   model_year    51525 non-null  int64  
     2   model         51525 non-null  object 
     3   condition     51525 non-null  object 
     4   cylinders     51525 non-null  int64  
     5   fuel          51525 non-null  object 
     6   odometer      51525 non-null  float64
     7   transmission  51525 non-null  object 
     8   type          51525 non-null  object 
     9   paint_color   51525 non-null  object 
     10  is_4wd        51525 non-null  bool   
     11  date_posted   51525 non-null  object 
     12  days_listed   51525 non-null  int64  
    dtypes: bool(1), float64(2), int64(3), object(7)
    memory usage: 4.8+ MB



```python
# Tambahkan nilai waktu dan tanggal pada saat iklan ditayangkan

# Merubah tipe data pada  column date_posted menjadi datetime
data['date_posted'] = pd.to_datetime(data['date_posted'], format='%Y-%m-%d %H:%M:%S', errors='raise')

# Menambahkan table hari, minggu, bulan, dan tahun iklan ditayangkan
data['days'] = data['date_posted'].dt.dayofweek 
data['week'] = data['date_posted'].dt.week 
data['month'] = data['date_posted'].dt.month 
data['year'] = data['date_posted'].dt.year 
```


```python
# menampilkan nilai waktu dan tanggal pada saat iklan ditayangkan

data.head()
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9400.0</td>
      <td>2011</td>
      <td>bmw x5</td>
      <td>good</td>
      <td>6</td>
      <td>gas</td>
      <td>145000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>others</td>
      <td>True</td>
      <td>2018-06-23</td>
      <td>19</td>
      <td>5</td>
      <td>25</td>
      <td>6</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25500.0</td>
      <td>2009</td>
      <td>ford f-150</td>
      <td>good</td>
      <td>6</td>
      <td>gas</td>
      <td>88705.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>white</td>
      <td>True</td>
      <td>2018-10-19</td>
      <td>50</td>
      <td>4</td>
      <td>42</td>
      <td>10</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5500.0</td>
      <td>2013</td>
      <td>hyundai sonata</td>
      <td>like new</td>
      <td>4</td>
      <td>gas</td>
      <td>110000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>red</td>
      <td>False</td>
      <td>2019-02-07</td>
      <td>79</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1500.0</td>
      <td>2003</td>
      <td>ford f-150</td>
      <td>fair</td>
      <td>8</td>
      <td>gas</td>
      <td>181613.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>others</td>
      <td>False</td>
      <td>2019-03-22</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>3</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14900.0</td>
      <td>2017</td>
      <td>chrysler 200</td>
      <td>excellent</td>
      <td>4</td>
      <td>gas</td>
      <td>80903.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>False</td>
      <td>2019-04-02</td>
      <td>28</td>
      <td>1</td>
      <td>14</td>
      <td>4</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Tambahkan usia kendaraan saat iklan ditayangkan

data['vehicle_age'] = (data['year'] - data_drop['model_year']) + 1

```


```python
# menampilkan statistik pada column usia kendaraan saat iklan ditayangkan

data['vehicle_age'].describe()
```




    count    29916.000000
    mean         9.560603
    std          6.286292
    min          1.000000
    25%          5.000000
    50%          8.000000
    75%         13.000000
    max        111.000000
    Name: vehicle_age, dtype: float64




```python
# Merubah nilai inf ke NaN dan merubah nilai NaN ke nilai 1 pada column vehicle_age

data['vehicle_age'] = pd.to_numeric(data['vehicle_age'], errors='coerce')
data['vehicle_age'] = data['vehicle_age'].replace(np.inf, int(float(0)))
data['vehicle_age'] = data['vehicle_age'].fillna(1).astype(int)
data['vehicle_age'] = data['vehicle_age'].astype('int64') 
```


```python
# Menampilkan usia kendaraan saat iklan ditayangkan

data.head()
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
      <th>vehicle_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9400.0</td>
      <td>2011</td>
      <td>bmw x5</td>
      <td>good</td>
      <td>6</td>
      <td>gas</td>
      <td>145000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>others</td>
      <td>True</td>
      <td>2018-06-23</td>
      <td>19</td>
      <td>5</td>
      <td>25</td>
      <td>6</td>
      <td>2018</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25500.0</td>
      <td>2009</td>
      <td>ford f-150</td>
      <td>good</td>
      <td>6</td>
      <td>gas</td>
      <td>88705.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>white</td>
      <td>True</td>
      <td>2018-10-19</td>
      <td>50</td>
      <td>4</td>
      <td>42</td>
      <td>10</td>
      <td>2018</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5500.0</td>
      <td>2013</td>
      <td>hyundai sonata</td>
      <td>like new</td>
      <td>4</td>
      <td>gas</td>
      <td>110000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>red</td>
      <td>False</td>
      <td>2019-02-07</td>
      <td>79</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>2019</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1500.0</td>
      <td>2003</td>
      <td>ford f-150</td>
      <td>fair</td>
      <td>8</td>
      <td>gas</td>
      <td>181613.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>others</td>
      <td>False</td>
      <td>2019-03-22</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>3</td>
      <td>2019</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14900.0</td>
      <td>2017</td>
      <td>chrysler 200</td>
      <td>excellent</td>
      <td>4</td>
      <td>gas</td>
      <td>80903.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>False</td>
      <td>2019-04-02</td>
      <td>28</td>
      <td>1</td>
      <td>14</td>
      <td>4</td>
      <td>2019</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Tambahkan jarak tempuh rata-rata kendaraan per tahun

data['avg_miles_year'] = (data['odometer'] / data['vehicle_age']) + 1
```


```python
# menampilkan statistik pada column jarak tempuh rata-rata per tahun

data['avg_miles_year'].describe()
```




    count     51525.000000
    mean      56672.249236
    std       60615.414651
    min           1.000000
    25%       11888.500000
    50%       20609.285714
    75%      104231.000000
    max      990001.000000
    Name: avg_miles_year, dtype: float64



Setelah kita menampilkan distribusi statistik pada colum jarak tempuh rata-rata per tahun dari kendaraan terdapat nilai NaN sebagai standar deviasi dan inf sebagai mean / rata-rata, hal ini dikarenakan pada column vehicle_age terdapat nilai 0 dimana pada saat menghitung jarak tempuh kendaraan 'odometer' dibagi dengan usia kendaraan 'vehicle_age' mengarah pada nilai NaN atau inf tersebut. sehingga untuk mengatasinya adalah dengan merubah nilai columns 'avg_miles_year' atau dengan mengkonversi inf dan NaN tersebut ke nilai 1 dengan tipe data interger, sehingga nilai inf dan NaN tersebut dapat digantikan dengan 1.



```python
# Merubah nilai inf ke NaN dan merubah nilai NaN ke nilai 1 pada tipe data interger

data['avg_miles_year'] = pd.to_numeric(data['avg_miles_year'], errors='coerce')
data['avg_miles_year'] = data['avg_miles_year'].replace(np.inf, int(float(0)))
data['avg_miles_year'] = data['avg_miles_year'].fillna(1).astype(int)
data['avg_miles_year'] = data['avg_miles_year'].astype('int64') 
```


```python
# menampilkan statistik pada column jarak tempuh rata-rata per tahun setelah di manipulasi

data['avg_miles_year'].describe()
```




    count     51525.000000
    mean      56672.042445
    std       60615.553841
    min           1.000000
    25%       11888.000000
    50%       20609.000000
    75%      104231.000000
    max      990001.000000
    Name: avg_miles_year, dtype: float64




```python
# Mungkin membantu untuk mengganti nilai pada kolom 'condition' dengan sesuatu yang dapat dimanipulasi dengan lebih mudah

data['condition'] = data['condition'].replace(['new', 'like new', 'excellent', 'good', 'fair', 'salvage'], [5, 4, 3, 2, 1, 0])
```


```python
# Menampilkan Dataset yang telah dimanipulasi 

data.head()
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
      <th>vehicle_age</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9400.0</td>
      <td>2011</td>
      <td>bmw x5</td>
      <td>2</td>
      <td>6</td>
      <td>gas</td>
      <td>145000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>others</td>
      <td>True</td>
      <td>2018-06-23</td>
      <td>19</td>
      <td>5</td>
      <td>25</td>
      <td>6</td>
      <td>2018</td>
      <td>1</td>
      <td>145001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25500.0</td>
      <td>2009</td>
      <td>ford f-150</td>
      <td>2</td>
      <td>6</td>
      <td>gas</td>
      <td>88705.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>white</td>
      <td>True</td>
      <td>2018-10-19</td>
      <td>50</td>
      <td>4</td>
      <td>42</td>
      <td>10</td>
      <td>2018</td>
      <td>1</td>
      <td>88706</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5500.0</td>
      <td>2013</td>
      <td>hyundai sonata</td>
      <td>4</td>
      <td>4</td>
      <td>gas</td>
      <td>110000.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>red</td>
      <td>False</td>
      <td>2019-02-07</td>
      <td>79</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>2019</td>
      <td>7</td>
      <td>15715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1500.0</td>
      <td>2003</td>
      <td>ford f-150</td>
      <td>1</td>
      <td>8</td>
      <td>gas</td>
      <td>181613.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>others</td>
      <td>False</td>
      <td>2019-03-22</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>3</td>
      <td>2019</td>
      <td>1</td>
      <td>181614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14900.0</td>
      <td>2017</td>
      <td>chrysler 200</td>
      <td>3</td>
      <td>4</td>
      <td>gas</td>
      <td>80903.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>black</td>
      <td>False</td>
      <td>2019-04-02</td>
      <td>28</td>
      <td>1</td>
      <td>14</td>
      <td>4</td>
      <td>2019</td>
      <td>3</td>
      <td>26968</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

Pada Poin ini kita berhasil menambahkan :

1. Hari, minggu, bulan, dan tahun kendaraan saat iklan ditayangkan
2. Usia kendaraan (dalam tahun) ketika iklan ditayangkan
3. Jarak tempuh rata-rata kendaraan per tahun 
4. Mengganti nilai string dengan skala numerik pada column condition


## Memeriksa Data yang Sudah Bersih


```python
# tampilkan informasi/rangkuman umum tentang DataFrame

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51525 entries, 0 to 51524
    Data columns (total 19 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   price           51525 non-null  float64       
     1   model_year      51525 non-null  int64         
     2   model           51525 non-null  object        
     3   condition       51525 non-null  int64         
     4   cylinders       51525 non-null  int64         
     5   fuel            51525 non-null  object        
     6   odometer        51525 non-null  float64       
     7   transmission    51525 non-null  object        
     8   type            51525 non-null  object        
     9   paint_color     51525 non-null  object        
     10  is_4wd          51525 non-null  bool          
     11  date_posted     51525 non-null  datetime64[ns]
     12  days_listed     51525 non-null  int64         
     13  days            51525 non-null  int64         
     14  week            51525 non-null  int64         
     15  month           51525 non-null  int64         
     16  year            51525 non-null  int64         
     17  vehicle_age     51525 non-null  int64         
     18  avg_miles_year  51525 non-null  int64         
    dtypes: bool(1), datetime64[ns](1), float64(2), int64(10), object(5)
    memory usage: 7.1+ MB



```python
# tampilkan sampel data

data.sample(frac=2, replace=True, random_state=1)

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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
      <th>vehicle_age</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33003</th>
      <td>4295.0</td>
      <td>2004</td>
      <td>chevrolet suburban</td>
      <td>2</td>
      <td>8</td>
      <td>gas</td>
      <td>129000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>others</td>
      <td>True</td>
      <td>2019-03-13</td>
      <td>22</td>
      <td>2</td>
      <td>11</td>
      <td>3</td>
      <td>2019</td>
      <td>1</td>
      <td>129001</td>
    </tr>
    <tr>
      <th>12172</th>
      <td>1.0</td>
      <td>2016</td>
      <td>ram 3500</td>
      <td>3</td>
      <td>10</td>
      <td>gas</td>
      <td>43200.0</td>
      <td>other</td>
      <td>truck</td>
      <td>white</td>
      <td>True</td>
      <td>2018-05-13</td>
      <td>24</td>
      <td>6</td>
      <td>19</td>
      <td>5</td>
      <td>2018</td>
      <td>3</td>
      <td>14401</td>
    </tr>
    <tr>
      <th>5192</th>
      <td>2999.0</td>
      <td>2004</td>
      <td>toyota sienna</td>
      <td>4</td>
      <td>6</td>
      <td>gas</td>
      <td>185696.0</td>
      <td>automatic</td>
      <td>mini-van</td>
      <td>grey</td>
      <td>False</td>
      <td>2019-02-26</td>
      <td>20</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>2019</td>
      <td>16</td>
      <td>11607</td>
    </tr>
    <tr>
      <th>32511</th>
      <td>15995.0</td>
      <td>2016</td>
      <td>chevrolet camaro lt coupe 2d</td>
      <td>4</td>
      <td>6</td>
      <td>gas</td>
      <td>30000.0</td>
      <td>manual</td>
      <td>coupe</td>
      <td>black</td>
      <td>False</td>
      <td>2018-07-09</td>
      <td>44</td>
      <td>0</td>
      <td>28</td>
      <td>7</td>
      <td>2018</td>
      <td>3</td>
      <td>10001</td>
    </tr>
    <tr>
      <th>50057</th>
      <td>5900.0</td>
      <td>2005</td>
      <td>toyota highlander</td>
      <td>2</td>
      <td>6</td>
      <td>gas</td>
      <td>195643.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>silver</td>
      <td>True</td>
      <td>2018-05-16</td>
      <td>26</td>
      <td>2</td>
      <td>20</td>
      <td>5</td>
      <td>2018</td>
      <td>14</td>
      <td>13975</td>
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
    </tr>
    <tr>
      <th>44639</th>
      <td>3900.0</td>
      <td>2003</td>
      <td>ford f250 super duty</td>
      <td>1</td>
      <td>8</td>
      <td>gas</td>
      <td>82300.0</td>
      <td>automatic</td>
      <td>truck</td>
      <td>white</td>
      <td>True</td>
      <td>2018-10-26</td>
      <td>94</td>
      <td>4</td>
      <td>43</td>
      <td>10</td>
      <td>2018</td>
      <td>1</td>
      <td>82301</td>
    </tr>
    <tr>
      <th>28674</th>
      <td>15500.0</td>
      <td>2007</td>
      <td>toyota tundra</td>
      <td>3</td>
      <td>8</td>
      <td>gas</td>
      <td>162000.0</td>
      <td>automatic</td>
      <td>offroad</td>
      <td>white</td>
      <td>True</td>
      <td>2018-11-02</td>
      <td>58</td>
      <td>4</td>
      <td>44</td>
      <td>11</td>
      <td>2018</td>
      <td>12</td>
      <td>13501</td>
    </tr>
    <tr>
      <th>36544</th>
      <td>7995.0</td>
      <td>2005</td>
      <td>chrysler 300</td>
      <td>2</td>
      <td>8</td>
      <td>gas</td>
      <td>183080.0</td>
      <td>automatic</td>
      <td>sedan</td>
      <td>grey</td>
      <td>False</td>
      <td>2018-12-24</td>
      <td>6</td>
      <td>0</td>
      <td>52</td>
      <td>12</td>
      <td>2018</td>
      <td>14</td>
      <td>13078</td>
    </tr>
    <tr>
      <th>4949</th>
      <td>16395.0</td>
      <td>2017</td>
      <td>ford escape</td>
      <td>3</td>
      <td>4</td>
      <td>gas</td>
      <td>104230.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>others</td>
      <td>True</td>
      <td>2018-06-21</td>
      <td>109</td>
      <td>3</td>
      <td>25</td>
      <td>6</td>
      <td>2018</td>
      <td>1</td>
      <td>104231</td>
    </tr>
    <tr>
      <th>10229</th>
      <td>2600.0</td>
      <td>1998</td>
      <td>toyota 4runner</td>
      <td>2</td>
      <td>6</td>
      <td>gas</td>
      <td>191000.0</td>
      <td>automatic</td>
      <td>SUV</td>
      <td>red</td>
      <td>False</td>
      <td>2019-04-01</td>
      <td>36</td>
      <td>0</td>
      <td>14</td>
      <td>4</td>
      <td>2019</td>
      <td>22</td>
      <td>8682</td>
    </tr>
  </tbody>
</table>
<p>103050 rows × 19 columns</p>
</div>



## Mempelajari Parameter Inti pada column :

- Harga
- Usia kendaraan ketika iklan ditayangkan
- Jarak tempuh
- Jumlah silinder
- Kondisi

Memeriksa korelasi pada column harga, usia kendaraan ketika iklan ditayangkan, Jarak tempuh, Jumlah silinder, Kondisi.


```python
# Menampilakan korelasi pada column harga, usia kendaraan ketika iklan ditayangkan, rata-rata umur jarak tempuh kendaraan pertahun, Jumlah silinder, Kondisi.

data[['price','model_year', 'condition', 'cylinders', 'vehicle_age','days_listed', 'avg_miles_year']].corr()

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
      <th>price</th>
      <th>model_year</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>vehicle_age</th>
      <th>days_listed</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>1.000000</td>
      <td>0.418347</td>
      <td>0.221518</td>
      <td>0.300342</td>
      <td>-0.242747</td>
      <td>-0.000682</td>
      <td>-0.120768</td>
    </tr>
    <tr>
      <th>model_year</th>
      <td>0.418347</td>
      <td>1.000000</td>
      <td>0.295711</td>
      <td>-0.142475</td>
      <td>-0.595968</td>
      <td>-0.005239</td>
      <td>-0.083427</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>0.221518</td>
      <td>0.295711</td>
      <td>1.000000</td>
      <td>-0.065661</td>
      <td>-0.158569</td>
      <td>-0.002404</td>
      <td>-0.132789</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>0.300342</td>
      <td>-0.142475</td>
      <td>-0.065661</td>
      <td>1.000000</td>
      <td>0.091976</td>
      <td>0.003181</td>
      <td>0.019955</td>
    </tr>
    <tr>
      <th>vehicle_age</th>
      <td>-0.242747</td>
      <td>-0.595968</td>
      <td>-0.158569</td>
      <td>0.091976</td>
      <td>1.000000</td>
      <td>0.005013</td>
      <td>-0.576436</td>
    </tr>
    <tr>
      <th>days_listed</th>
      <td>-0.000682</td>
      <td>-0.005239</td>
      <td>-0.002404</td>
      <td>0.003181</td>
      <td>0.005013</td>
      <td>1.000000</td>
      <td>-0.003438</td>
    </tr>
    <tr>
      <th>avg_miles_year</th>
      <td>-0.120768</td>
      <td>-0.083427</td>
      <td>-0.132789</td>
      <td>0.019955</td>
      <td>-0.576436</td>
      <td>-0.003438</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<div class="alert alert-danger">
<b>Reviewer's comment v1</b> <a class="tocSkip"></a>

Daripada menggunakan `odometer` sebagai parameter inti, `avg_mileage` akan lebih baik karena mempertimbangkan usia kendaraan sehingga perbandingan yang dilakukan akan lebih fair.

<div class="alert alert-block alert-info">
<b> Parameter inti sudah diganti menggunakan avg_miles.</b> <a class="tocSkip"></a>
</div>


```python
# Menampilkan korelasi pada column harga, usia kendaraan ketika iklan ditayangkan, rata-rata umur jarak tempuh kendaraan pertahun, Jumlah silinder, Kondisi dengan menggunakan scatter matrix

data_corr = data[['price','model_year', 'condition', 'cylinders', 'vehicle_age','days_listed', 'avg_miles_year']]
pd.plotting.scatter_matrix(data_corr, figsize=(20, 20)) 
plt.suptitle('korelasi dengan parameter menggunakan scatter matrix', y=1.05);
plt.show()
```


    
![png](output_89_0.png)
    


Disini kita dapat melihat sel dari grid yang menunjukkan korelasi / hubungan antar column : 

1. sel kiri pertama menampilkan hubungan antara harga dan harga, pada cel histogramnya menunjukkan distribusi statistik haraga paling tinggi adalah diatas 300000, untuk detailnya akan dilihat pada bagian selanjutnya.

2. sel kiri kedua menampilkan hubungan antara harga dan rata-rata umur jarak tempuh kendaraan pertahun dimana titik scatter tersebar dengan padat diantara 0 - 4 tahun.

3. sel kiri ketiga menampilkan hubungan antara harga dan rata-rata umur kendaraan dimana titik scatter tersebar dengan padat diantara 0.0 - 0.2.

4. sel kiri keempat menampilkan hubungan harga dan jumlah silinder dimana titik scatter menunjukkan jumlah silinder 6 - 8 lebih banyak dipilih dibandingkan dengan jumlah silinder yang lain.

5. sel kiri kelima menampilka hubungan harga dengan kondisi kendaraan dimana titik scatter menunjukkan kondisi kendaraan dengan harga  5 = new, 4= like new, 3=excelent, 2=good lebih banyak dipilih dibandingkan dengan kondisi 1=fair, 0=salvage.  


```python
# Menampilkan korelasi pada column harga, usia kendaraan ketika iklan ditayangkan, rata-rata umur jarak tempuh kendaraan, Jumlah silinder, Kondisi dengan menggunakan Histogram 

data_corr.hist(bins=30, figsize=(15, 10))
plt.suptitle('korelasi dengan parameter menggunakan histogram', y=1.05);
plt.show()

```


    
![png](output_91_0.png)
    


Mari kita lihat parameter dengan lebih detail


```python
# Menampilkan histogram dari harga dengan detail
data['price'].hist(bins=100, range=[0, 80000])

# menambahkan judul dan nama sumbu 
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari harga kendaraan");
```


    
![png](output_93_0.png)
    


Histogram diatas menunjukkan sebaran data dalam colum harga, histogram ini menunjukkan penyebaran menanjak dari kiri dan menurun di kanan, dimana sebagian besar mobil jatuh pada harga rendah - menengah. Histogram tertinggi berada pada frekuensi diatas 2500. frekunsi atau sebaran data berkisar dari 1 hingga 375000 dengan median 9000. dapat disimpulkan data pada column price memiliki outlier yang signifikan.


```python
## Menampilkan histogram dari umur kendaraan dengan detail
data['model_year'].hist(bins=50, range=[1900, 2019])

# menambahkan judul dan nama sumbu 
plt.xlabel('model_year')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari tahun model kendaraan ");
```


    
![png](output_95_0.png)
    


Histogram diatas menunjukkan data dalam column model_year dimana data tersebara dari kiri kekanan dengan nilai puncak diatas 12000 dengan rentang nilai 2000 - 2019. 


```python
# Menampilkan histogram dari kondisi dengan detail
data['condition'].hist(bins=40)

# menambahkan judul dan nama sumbu 
plt.xlabel('condition')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari kondisi kendaraan");
```


    
![png](output_97_0.png)
    


Histogram diatas menunjukkan sebaran data dalam column condition dimana data tegak lurus dengan nilai terbanyak pada colum 2 = good, 3= excellent. dengan puncak tertinggi diatas 20000 dengan rentang nilai dari 0 = salvage, 1= fair, 4= like new, 5= new. disini kita dapat menyimpulkan bahwa distribusi memiliki beberapa outlier. 


```python
# Menampilkan histogram dari cylinders dengan detail
data['cylinders'].hist(bins= 9)

# menambahkan judul dan nama sumbu 
plt.xlabel('cylinders')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari silinder kendaraan");

```


    
![png](output_99_0.png)
    


Histogram diatas menunjukkan sebaran data dalam column cylinders dimana data tegak lurus dengan nilai terbanyak pada colum 4, 6, 8. dengan puncak tertinggi diatas 14000. Faktanya sebagian besar kendaraan dalam data mewakili kendaraan yang dibuat dengan 4 silinder, 6 silinder, dan 8 silinder . dapat disimpulkan data pada column jarak tempuh terdapat outlier dimana nilai median dan meannya adalah 6 dan standar deviasinya adalah 1.6.


```python
## Menampilkan histogram dari umur kendaraan dengan detail
data['vehicle_age'].hist(bins=20, range=[0, 111])

# menambahkan judul dan nama sumbu 
plt.xlabel('vehicle_age')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari umur kendaraan");
```


    
![png](output_101_0.png)
    


pada histogram diatas menunjukkan sebaran data pada column umur kendaraan dimana data tersebar dari 1 - 20, dimana nilai puncak nya pada 30000


```python
## Menampilkan histogram dari jumlah hari iklan ditayangkan hingga dihapus
data['days_listed'].hist(bins=20, range=[0,  271])

# menambahkan judul dan nama sumbu 
plt.xlabel('days_listed')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari jumlah hari iklan ditayangkan hingga dihapus");
```


    
![png](output_103_0.png)
    


Pada histogram diatas menunjukkan sebaran data dari jumlah hari iklan ditayangkan hingga dihapus dimana rentang nilai dari 0 hari sampai 180, dimana nilai puncak terdapat pada rentang 0 - 50 hari. 


```python
# Menampilkan histogram dari jarak tempuh kendaraan pertahun
data['avg_miles_year'].hist(bins=50, range=[0, 70])

# menambahkan judul dan nama sumbu 
plt.xlabel('avg_miles_year')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari rata-rata jarak tempuh kendaraan pertahun");
```


    
![png](output_105_0.png)
    


Histogram diatas menunjukkan sebaran data dalam column avg_miles_year dimana data tersebar dari kiri kekanan dengan nilai terbanyak pada nilai 0, dengan puncak tertinggi diatas 200 dengan rentang nilai dari 0 - 700. dapat disimpulkan data pada column rata-rata jarak tempuh kendaraan pertahun terdapat outlier dimana nilai maksimalnya adalah 200. 


```python
# distribusi statistik pada data yang belum filter

data[['price','model_year', 'condition', 'cylinders', 'vehicle_age','days_listed', 'avg_miles_year']].describe()
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
      <th>price</th>
      <th>model_year</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>vehicle_age</th>
      <th>days_listed</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.00000</td>
      <td>51525.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12132.464920</td>
      <td>2009.816419</td>
      <td>2.637535</td>
      <td>6.121067</td>
      <td>5.970383</td>
      <td>39.55476</td>
      <td>56672.042445</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10040.803015</td>
      <td>6.091605</td>
      <td>0.712447</td>
      <td>1.657457</td>
      <td>6.386633</td>
      <td>28.20427</td>
      <td>60615.553841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1908.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5000.000000</td>
      <td>2007.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>19.00000</td>
      <td>11888.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9000.000000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>33.00000</td>
      <td>20609.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16839.000000</td>
      <td>2014.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>53.00000</td>
      <td>104231.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>375000.000000</td>
      <td>2019.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>111.000000</td>
      <td>271.00000</td>
      <td>990001.000000</td>
    </tr>
  </tbody>
</table>
</div>



Dari visualisasi awal untuk mendeteksi outlier, kita dapat mengamati bahwa variabel harga, dan rata-rata jarak tempuh kendaraan pertahian memiliki outlier yang signifikan. Melihat variabel harga, mudah untuk mendeteksi outlier yang dihasilkan dari data yang salah. Harga minimum adalah 1 dan harga maksimum adalah 375.000. Demikian pula pada column, usia kendaraan dimana niai rata-ratanya adalah 6 tahun, sedangkan usia kendaraan maksimum adalah 111 tahun dimana standar deviasinya 6 tahun jelas terdapat outlier dalam variabel ini. Kita dapat melihat bahwa ada begitu banyak outlier dalam data sehingga kita harus menanganinya sebelum kita dapat melanjutkan dengan analisis.

## Mempelajari dan Menangani Outlier


Dengan menggunakan metode IQR  ukuran variabilitas yang didasarkan pada pembagian kumpulan data menjadi kuartil. Kuartil membagi kumpulan data terurut menjadi empat bagian yang sama besar. Nilai yang memisahkan bagian-bagian ini disebut kuartil pertama, kedua (median), dan ketiga yang masing-masing dilambangkan dengan Q1, Q2, dan Q3

dengan menggunakan metode ini kita dapat mendeteksi nilai atas dan bawah dari outlier serta memfilter data yang teridentifikasi berisi outlier. 

![image-2.png](attachment:image-2.png)


```python
# fungsi menentukan batas bawah outlier

def lower_whisker(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    
    iqr = q3 - q1
    
    return q1 - 1.5 * iqr
```


```python
# fungsi menentukan batas atas outlier

def upper_whisker(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    
    iqr = q3 - q1
    
    return q1 + 1.5 * iqr
```


```python
# batas bawah column price 

lower_price = lower_whisker(data, 'price')
lower_price
```




    -12758.5




```python
# batas atas column price 

upper_price = upper_whisker(data, 'price')
upper_price
```




    22758.5




```python
# batas bawah column 'vehicle_age' 

lower_age = lower_whisker(data, 'vehicle_age')
lower_age
```




    -12.5




```python
# batas atas column 'vehicle_age'

upper_age = upper_whisker(data, 'vehicle_age')
upper_age
```




    14.5




```python
# batas bawah column 'odometer' 

lower_odometer = lower_whisker(data, 'odometer')
lower_odometer
```




    -32561.5




```python
# batas atas column 'odometer'

upper_odometer = upper_whisker(data, 'odometer')
upper_odometer
```




    183897.5




```python
# 
data_clean = data.query('(price > @lower_price and price < @upper_price) and (vehicle_age > @lower_age and vehicle_age < @upper_age) and (odometer > @lower_odometer and odometer < @upper_odometer)')
```


```python
# distribusi statistik yang telah di filter nilai outliernya

data_clean.describe()
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
      <th>price</th>
      <th>model_year</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>odometer</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
      <th>vehicle_age</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.00000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9897.044005</td>
      <td>2010.629763</td>
      <td>2.683778</td>
      <td>5.791866</td>
      <td>105759.028876</td>
      <td>39.55681</td>
      <td>3.006674</td>
      <td>26.927647</td>
      <td>6.642140</td>
      <td>2018.303730</td>
      <td>4.509069</td>
      <td>60286.473334</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5596.726017</td>
      <td>4.708463</td>
      <td>0.673607</td>
      <td>1.657915</td>
      <td>41784.646569</td>
      <td>28.37710</td>
      <td>1.995006</td>
      <td>15.108158</td>
      <td>3.467112</td>
      <td>0.459874</td>
      <td>4.215061</td>
      <td>54193.876295</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1908.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2018.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5500.000000</td>
      <td>2008.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>80000.000000</td>
      <td>19.00000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>2018.000000</td>
      <td>1.000000</td>
      <td>13047.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8988.000000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>105669.000000</td>
      <td>33.00000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>7.000000</td>
      <td>2018.000000</td>
      <td>2.000000</td>
      <td>26312.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13995.000000</td>
      <td>2014.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>131861.500000</td>
      <td>53.00000</td>
      <td>5.000000</td>
      <td>40.000000</td>
      <td>10.000000</td>
      <td>2019.000000</td>
      <td>8.000000</td>
      <td>104231.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22745.000000</td>
      <td>2019.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>183804.000000</td>
      <td>271.00000</td>
      <td>6.000000</td>
      <td>52.000000</td>
      <td>12.000000</td>
      <td>2019.000000</td>
      <td>14.000000</td>
      <td>183764.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# distribusi statistik yang telah belum di filter nilai outliernya

data.describe()
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
      <th>price</th>
      <th>model_year</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>odometer</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
      <th>vehicle_age</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.00000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
      <td>51525.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12132.464920</td>
      <td>2009.816419</td>
      <td>2.637535</td>
      <td>6.121067</td>
      <td>115199.014508</td>
      <td>39.55476</td>
      <td>3.005434</td>
      <td>26.873498</td>
      <td>6.628491</td>
      <td>2018.307462</td>
      <td>5.970383</td>
      <td>56672.042445</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10040.803015</td>
      <td>6.091605</td>
      <td>0.712447</td>
      <td>1.657457</td>
      <td>60484.863376</td>
      <td>28.20427</td>
      <td>1.997759</td>
      <td>15.138854</td>
      <td>3.474134</td>
      <td>0.461447</td>
      <td>6.386633</td>
      <td>60615.553841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1908.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2018.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5000.000000</td>
      <td>2007.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>75668.000000</td>
      <td>19.00000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>2018.000000</td>
      <td>1.000000</td>
      <td>11888.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9000.000000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>110908.000000</td>
      <td>33.00000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>7.000000</td>
      <td>2018.000000</td>
      <td>3.000000</td>
      <td>20609.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16839.000000</td>
      <td>2014.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>147821.000000</td>
      <td>53.00000</td>
      <td>5.000000</td>
      <td>40.000000</td>
      <td>10.000000</td>
      <td>2019.000000</td>
      <td>10.000000</td>
      <td>104231.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>375000.000000</td>
      <td>2019.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>990000.000000</td>
      <td>271.00000</td>
      <td>6.000000</td>
      <td>52.000000</td>
      <td>12.000000</td>
      <td>2019.000000</td>
      <td>111.000000</td>
      <td>990001.000000</td>
    </tr>
  </tbody>
</table>
</div>



Setelah menentukan nilai atas dan bawah dari data outlier dapat dilihat pada distribusi statistik pada column price dimna nilai  min nya sebesar 12774 dan nilai maks 22745, serta column model year dimana nilai min 1997 dan nilai maks 2017. dll.

## Mempelajari Parameter Inti Tanpa Outlier

Membuat grafik histogram dengan data yang telah difilter dan membandingkan dengan histogram yang masih memiliki nilai yang outlier serta kesimpulan masing-masing value perbandingan valuenya.


```python
# Histogram dengan value yang telah difilter 

data_clean[['price', 'model_year', 'vehicle_age','days_listed', 'avg_miles_year']].hist(bins=100, figsize=(15, 10))
plt.suptitle('Histogram dengan distribusi statistik data yang telah di filter', y=0.95);
```


    
![png](output_125_0.png)
    



```python
# Histogram dengan value yang masih memiliki nilai outlier 

data[['price', 'model_year', 'vehicle_age','days_listed', 'avg_miles_year']].hist(bins=100, figsize=(15, 10))
plt.suptitle('Histogram dengan distribusi statistik data yang masih memiliki nilai outlier', y=0.95);
```


    
![png](output_126_0.png)
    


Kesimpulan :

1. Dengan menggunakan metode statistik IQR untuk menentukan distribusi nilai baku untuk menentukan rentang data normal dimana nilai nya dibagi menjadi 3 kuartil yakni q1 untuk nilai bawah , q2 untuk median, dan q3 untuk batas atas dari nilai outlier nya. sehingga sebaran data menjadi normal. 

2. Dapat dilihat dari sebaran data pada perbandingan histogram diatas, histogram yang telah di tentukan upper dan lower outliernya pada column price dimana sebaran datanya 5000 hingga 22745, dan data yang belum di tentukan nilai upper dan lower outliernya pada column price dimana sebaran datanya terbentang dari 0 hingga 375000. 


## Masa Berlaku Iklan

Analisis berapa hari iklan ditayangkan pada column days_listed


```python
# Ditribusi statistik data pada column days_listed
data_clean['days_listed'].describe()
```




    count    35064.00000
    mean        39.55681
    std         28.37710
    min          0.00000
    25%         19.00000
    50%         33.00000
    75%         53.00000
    max        271.00000
    Name: days_listed, dtype: float64




```python
# Tabel Histogram dari iklan ditayangkan pada column days listed
data_clean['days_listed'].hist(bins=100)

# menambahkan judul dan nama sumbu 
plt.xlabel('days_listed')
plt.ylabel('Frequency')
plt.title("Tabel Histogram dari iklan ditayangkan");
```


    
![png](output_130_0.png)
    



```python
# Melihat nilai mean dan median pada column days_listed

print('Nilai mean saat iklan ditayangkan adalah {:.2f} hari dan mediannya adalah {:.1f} hari'
      .format(data_clean['days_listed'].mean(), data_clean['days_listed'].median()))
```

    Nilai mean saat iklan ditayangkan adalah 39.56 hari dan mediannya adalah 33.0 hari


Kesimpulan :

1. Histogram di atas menunjukkan distribusi rentang nilai dari iklan pada kendaraan, dimana rentang nilai yang tersebar dengan nilai minimun 0 hari dan maksimum 271 hari setelah nilai outliernya di filter.  

2. kita dapat melihat pada column diatas bahwa rata-rata waktu iklan ditayangkan adalah 39 hari dan mediannya adalah 33 hari.

3. Secara umum jika kita melihat nilai pada median dan mean dimana pada umumnya iklan ditayangkan sekitar 1 bulan yakni di rentang nilai 33 hari dan 39 hari. 

4. Untuk memastikan rentang nilai minimum dan nilai maksimum pada column days_listed, sebaiknya kita mengidentifikasi dengan melakukan filter data berdasarkan column dengan nilai-nilai yang lebih memungkinkan dalam penarikan kesimpulan.


```python
# mengidentifikasi jumlah rata-rata hari iklan ditayangkan hingga dihapus dengan harga kendaraan 

ads_price = data_clean.pivot_table(index='days_listed', values='price', aggfunc=['mean', 'median', 'count'])
print(ads_price)
```

                         mean   median count
                        price    price price
    days_listed                             
    0            10649.093750  10247.5    32
    1            10168.482759   8997.0   116
    2            10655.735955   9475.0   178
    3            10191.481343   9599.5   268
    4             9790.897163   8500.0   282
    ...                   ...      ...   ...
    252          11500.000000  11500.0     1
    256           8980.000000   8980.0     1
    261           3800.000000   3800.0     1
    267           5500.000000   5500.0     1
    271           5200.000000   5200.0     1
    
    [219 rows x 3 columns]



```python
# mengidentifikasi jumlah rata-rata hari iklan ditayangkan hingga dihapus dengan model kendaraan 

ads_modelCar = data_clean.pivot_table(index='days_listed', values='model_year', aggfunc=['mean', 'median', 'count'])
print(ads_modelCar)
```

                        mean     median      count
                  model_year model_year model_year
    days_listed                                   
    0            2010.656250     2010.5         32
    1            2010.905172     2012.0        116
    2            2011.286517     2012.0        178
    3            2010.869403     2011.0        268
    4            2009.631206     2010.5        282
    ...                  ...        ...        ...
    252          2017.000000     2017.0          1
    256          2012.000000     2012.0          1
    261          2012.000000     2012.0          1
    267          2011.000000     2011.0          1
    271          2011.000000     2011.0          1
    
    [219 rows x 3 columns]


Kesimpulan : 

1. Dengan mengidentifikasi jumlah rata-rata hari iklan ditayangkan hingga dihapus dengan harga kendaraan dan model year, dapat ditarik kesimpulan bahwa batas tercepat iklan dihapus pada 0 hari dan iklan terlama di tayangkan pada 271 hari. dengan hipotesis ini mari kita filter dengan metode pengindeksaan boleean yakni query. dengan menampilkan iklan dengan waktu tercepat dan iklan dengan waktu yang tercepat.


```python
# Menampilkan iklan dengan waktu pada rata - rata

print('Jumlah iklan dengan waktu tercepat : {:.1f} '.format(len(data_clean.query('days_listed > 0 and days_listed < 39'))))
```

    Jumlah iklan dengan waktu tercepat : 20373.0 



```python
# Menampilkan iklan dengan waktu tercepat

print('Jumlah iklan dengan waktu tercepat : {:.1f} '.format(len(data_clean.query('days_listed == 0'))))
```

    Jumlah iklan dengan waktu tercepat : 32.0 



```python
# Menampilkan iklan dengan waktu terlama

print('Jumlah iklan dengan waktu terlama : {:.1f} '.format(len(data_clean.query('days_listed == 271'))))
```

    Jumlah iklan dengan waktu terlama : 1.0 


Kesimpulan :

1. Dari analisa jumlah hari iklan ditampilkan, kita dapat melihat bahwa waktu iklan di tampilkan pada dengan menggunakan mean pada column days_listed dengan jumlah 20373. iklan dengan waktu tercepat didapat denga mefilter data dimana kendaraan yang laku pada waktu kurang dari 1 hari dengan jumlah 32 kemungkinan ini dapat terjadi dikarenakan kemungkinan kendaraan terjual pada hari yang sama dan kemungkinan lainnya. untuk iklan dengan waktu terlama dengan memfilter iklan yang waktunya lebih dari 271 hari berjumlah 1. 


## Harga Rata-Rata Setiap Jenis Kendaraan

Grafik Analisis Jumlah iklan dan harga rata-rata setiap jenis kendaraan, serta ketergantungan jumlah iklan pada jenis kendaraan. berdasarkan 2 jenis kendaraan dengan jumlah iklan yang paling banyak.


```python
# filter data berdasarkan jumlah iklan dengan harga rata-rata kendaraan 

avg_priceType = data_clean.groupby(['type']).agg({'price' : 'mean', 'days_listed' : 'count'}).sort_values(by=['days_listed'], ascending=False).reset_index().head(5)
avg_priceType
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
      <th>type</th>
      <th>price</th>
      <th>days_listed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sedan</td>
      <td>7353.020677</td>
      <td>10640</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SUV</td>
      <td>9978.551449</td>
      <td>9009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>truck</td>
      <td>12890.936043</td>
      <td>6520</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pickup</td>
      <td>12720.132534</td>
      <td>3501</td>
    </tr>
    <tr>
      <th>4</th>
      <td>coupe</td>
      <td>10814.170497</td>
      <td>1349</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Grafik hex berdasarkan ketergantungan jumlah hari iklan ditayangkan dengan harga rata-rata untuk setiap jenis kendaraan.

avg_priceType.plot(x='price', y='days_listed', title='Hexagonal satuan dan jarak iklan dan harga ', 
               kind='hexbin', gridsize=20, figsize=(8,6), sharex=False, grid=True
);
```


    
![png](output_143_0.png)
    


Kesimpulan :    

1. Pada diagram hexagonal diatas terdapat hubungan negatif yang rendah dikarenakan titik yang tersebar menjauhi garis lurus pada Analisis Jumlah iklan dan harga rata-rata setiap jenis kendaraan dimana puncak tertinggi sumbu = y iklan ditayangkan adalah lebih dari 10000 pada iklan yang ditayangkan  dari rentang sumbu = x data harga 0 sd 8000. 

2. Setelah menganalisa keterkaitan jumlah iklan dan rata-rata harga untuk setiap jenis kendaraan, dari hasil analisis jumlah hari iklan ditampilkan dimana dengan mengelompokkan data yang telah dihilangkan outliernya lalu melihat rata-rata dari harga dan jumlah hari dimana kendaraan ditampilkan,  maka didapatlah hasil dengan jenis kendaraan pada jumlah iklan terbanyak berdasarkan harga rata-rata ialah SUV dan Sedan. selanjutnya untuk menemukan keterkaitan / korelasi pada data untuk membantu menjawab pertanyaan bisnis dalam menentukan faktor-faktor apa saja yang mempengaruhi harga kendaraan. 

## Faktor Harga

Menganalisa jenis kendaraan dengan jumlah rata-rata harga terbanyak apakah terdapat keterkaitan / korelasi pada column: 
1.   Usia = vehicle_age (numerical) scatterplot
2.   Jarak Tempuh = odometer (numerical) scatterplot
3.   Kondisi = condition (kategorical) boxplot
4.   Tipe Transmisi = transmission (kategorical) boxplot
5.   Warna Kendaraan = paint color (kategorical) boxplot






```python
# Membuat Matriks korelasi d
corrMatriks = data_clean.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corrMatriks, annot=True)
plt.title('Correlation Matrix Plot')
plt.show()
```


    
![png](output_147_0.png)
    


analisis awal dapat dilihat dari grafik korelasi matrix diatas: 

1. faktor yang mempengaruhi harga kendaraan dengan korelasi positif yang lemah adalah model_year dengan nilai sebesar 0.54, condition dengan nilai 0.12, cylinders 0.26, is_4wd 0.27  

3. adapun faktor lain yang korelasinya negatif adalah days_listed, vehicle age, avg_miles_year.

Langkah selanjunya : kita akan mengelompokkan jenis kendaraan yang populer untuk menganalisa apakah harga mempunyai keterkaitan / korelasi pada usia, jarak tempuh, kondisi, tipe transmisi, dan warnanya. (tipe data kategorical)


```python
# Mengelompokkan jenis kendaraan dengan harga kendaraan

(data_clean.groupby('type')['price'].mean().sort_values().plot(kind='bar', title='Jenis kendaraan dengan harga terbanyak'));
```


    
![png](output_149_0.png)
    



```python
# Menampilkan distribusi data statistik pada jenis kendaraan dengan harga kendaraan 

vehicle_type = data_clean.groupby('type')['price'].describe().sort_values('mean', ascending=False).reset_index()
vehicle_type
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
      <th>type</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>truck</td>
      <td>6520.0</td>
      <td>12890.936043</td>
      <td>5936.632797</td>
      <td>1.0</td>
      <td>8500.0</td>
      <td>13900.0</td>
      <td>17900.00</td>
      <td>22745.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pickup</td>
      <td>3501.0</td>
      <td>12720.132534</td>
      <td>5885.569218</td>
      <td>1.0</td>
      <td>7995.0</td>
      <td>12900.0</td>
      <td>17900.00</td>
      <td>22700.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>offroad</td>
      <td>100.0</td>
      <td>12418.410000</td>
      <td>5650.142451</td>
      <td>15.0</td>
      <td>8475.0</td>
      <td>12500.0</td>
      <td>16923.75</td>
      <td>22000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>convertible</td>
      <td>231.0</td>
      <td>11340.255411</td>
      <td>5791.299277</td>
      <td>1.0</td>
      <td>6993.0</td>
      <td>11000.0</td>
      <td>15997.50</td>
      <td>22300.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>coupe</td>
      <td>1349.0</td>
      <td>10814.170497</td>
      <td>6650.249186</td>
      <td>1.0</td>
      <td>4995.0</td>
      <td>9990.0</td>
      <td>17300.00</td>
      <td>22500.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SUV</td>
      <td>9009.0</td>
      <td>9978.551449</td>
      <td>5368.963135</td>
      <td>1.0</td>
      <td>5990.0</td>
      <td>8995.0</td>
      <td>13995.00</td>
      <td>22692.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>wagon</td>
      <td>1223.0</td>
      <td>9544.427637</td>
      <td>4764.530154</td>
      <td>188.0</td>
      <td>5992.5</td>
      <td>8400.0</td>
      <td>12995.00</td>
      <td>22490.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>other</td>
      <td>183.0</td>
      <td>9526.163934</td>
      <td>4284.225057</td>
      <td>1000.0</td>
      <td>6000.0</td>
      <td>8995.0</td>
      <td>12149.00</td>
      <td>22500.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bus</td>
      <td>4.0</td>
      <td>8975.000000</td>
      <td>1495.270321</td>
      <td>7500.0</td>
      <td>7800.0</td>
      <td>8950.0</td>
      <td>10125.00</td>
      <td>10500.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>van</td>
      <td>464.0</td>
      <td>8829.614224</td>
      <td>4780.662625</td>
      <td>1.0</td>
      <td>5699.0</td>
      <td>7995.0</td>
      <td>11405.00</td>
      <td>21805.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>mini-van</td>
      <td>932.0</td>
      <td>8421.343348</td>
      <td>4697.184700</td>
      <td>1.0</td>
      <td>4995.0</td>
      <td>7483.5</td>
      <td>10995.00</td>
      <td>21999.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sedan</td>
      <td>10640.0</td>
      <td>7353.020677</td>
      <td>4049.911264</td>
      <td>1.0</td>
      <td>4500.0</td>
      <td>6550.0</td>
      <td>9888.50</td>
      <td>22199.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>hatchback</td>
      <td>908.0</td>
      <td>7122.843612</td>
      <td>3813.504634</td>
      <td>1.0</td>
      <td>4600.0</td>
      <td>6495.0</td>
      <td>8950.00</td>
      <td>21000.0</td>
    </tr>
  </tbody>
</table>
</div>



Pada tahap ini saya mengelompokkan jenis kendaraan dengan rata-rata harga, dengan mengambil sampel 5 jenis kendaraan dengan harga rata-rata terbanyak maka disini kita akan membuat variable baru dengan 5 jenis kendaraan populer yakni : truck, pickup, offroad, convertible, coupe. 


```python
# Memuat dan menampilkan variable baru dengan jenis kendaraan dengan rata-rata harga terbanyak

most_type5_pric = data_clean[data_clean.type.isin(['bus', 'truck', 'pickup', 'offroad', 'coupe'])]
most_type5_pric.head()
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
      <th>price</th>
      <th>model_year</th>
      <th>model</th>
      <th>condition</th>
      <th>cylinders</th>
      <th>fuel</th>
      <th>odometer</th>
      <th>transmission</th>
      <th>type</th>
      <th>paint_color</th>
      <th>is_4wd</th>
      <th>date_posted</th>
      <th>days_listed</th>
      <th>days</th>
      <th>week</th>
      <th>month</th>
      <th>year</th>
      <th>vehicle_age</th>
      <th>avg_miles_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1500.0</td>
      <td>2003</td>
      <td>ford f-150</td>
      <td>1</td>
      <td>8</td>
      <td>gas</td>
      <td>181613.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>others</td>
      <td>False</td>
      <td>2019-03-22</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>3</td>
      <td>2019</td>
      <td>1</td>
      <td>181614</td>
    </tr>
    <tr>
      <th>10</th>
      <td>19500.0</td>
      <td>2011</td>
      <td>chevrolet silverado 1500</td>
      <td>3</td>
      <td>8</td>
      <td>gas</td>
      <td>128413.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>black</td>
      <td>True</td>
      <td>2018-09-17</td>
      <td>38</td>
      <td>0</td>
      <td>38</td>
      <td>9</td>
      <td>2018</td>
      <td>8</td>
      <td>16052</td>
    </tr>
    <tr>
      <th>12</th>
      <td>18990.0</td>
      <td>2012</td>
      <td>ram 1500</td>
      <td>3</td>
      <td>8</td>
      <td>gas</td>
      <td>140742.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>others</td>
      <td>True</td>
      <td>2019-04-02</td>
      <td>37</td>
      <td>1</td>
      <td>14</td>
      <td>4</td>
      <td>2019</td>
      <td>1</td>
      <td>140743</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17990.0</td>
      <td>2013</td>
      <td>ram 1500</td>
      <td>3</td>
      <td>8</td>
      <td>gas</td>
      <td>104230.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>red</td>
      <td>True</td>
      <td>2018-05-15</td>
      <td>111</td>
      <td>1</td>
      <td>20</td>
      <td>5</td>
      <td>2018</td>
      <td>1</td>
      <td>104231</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14990.0</td>
      <td>2010</td>
      <td>ram 1500</td>
      <td>3</td>
      <td>8</td>
      <td>gas</td>
      <td>130725.0</td>
      <td>automatic</td>
      <td>pickup</td>
      <td>red</td>
      <td>True</td>
      <td>2018-12-30</td>
      <td>13</td>
      <td>6</td>
      <td>52</td>
      <td>12</td>
      <td>2018</td>
      <td>9</td>
      <td>14526</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menampilkan persentasi dari jenis kendaraan dengan jumlah harga terbanyak 

(most_type5_pric.pivot_table(index='type', values='price', aggfunc=['count', 'mean'])
     .plot(y='mean', kind='pie', 
           title = 'persentasi pie chart dari 5 jenis kendaraan dengan jumlah harga terbanyak ', 
           figsize=(8, 8), autopct='%1.1f%%', shadow=True)
);
```


    
![png](output_153_0.png)
    


pada grafik pie chart ini dapat diketahui bahwa jenis kendaraan dengan rata-rata jumlah harga terbanyak terdapat pada jenis kendaraan truck sebesar 22.3%, diikuti dengan jenis kendaraan pickup sebesar 22.0%


```python
# Korelasi 5 jenis kendaraan dengan rata-rata harga terbanyak dan usia kendaraan 

most_type5_pric.plot.scatter(x='vehicle_age', y='price', title='Scatterplot berdasarkan korelasi usia kendaraan terhadap harga', alpha=0.05);
```


    
![png](output_155_0.png)
    


Kesimpulan : dari hasil analisa korelasi pada scatterplot diatas terdapat korelasi negatif pada keterkaitan antara harga dengan usia kendaraan dimana pencaran titik semakin melebar. serta semakin tua kendaraan maka semakin rendah harganya sementara mobil baru diantara kurang dari lima tahun memiliki harga yang tinggi.


```python
# Korelasi  jenis kendaraan dengan rata-rata harga terbanyak dengan jarak tempuh  

most_type5_pric.plot.scatter(x='odometer', y='price', title='Scatterplot berdasarkan korelasi jarak tempuh terhadap harga', alpha=0.05);
```


    
![png](output_157_0.png)
    


Kesimpulan : dari hasil analisa pada scatterplott diatas terdapat korelasi negatif dimana titik tersebar dari atas kiri ke kanan bawah, dimana titik berkonsetrasi padat pada nilai odometer diantara 100000 dan 125000.


```python
 # Korelasi  jenis kendaraan dengan rata-rata harga terbanyak dengan kondisi kendaraan dengan lebih 50 iklan ditampilkan.

most_type5_pric.plot.scatter(x='condition', y='price', title='Scatterplot berdasarkan korelasi kondisi terhadap harga', alpha=0.05);    

```


    
![png](output_159_0.png)
    


Kesimpulan pada grafik scatter diatas terhadap kondisi kendaraan terhadap harga terdapat korelasi positif lemah dimana kemiringan titik dari bawah kiri ke atas kanan, dimana konsentrasi titik yang padat pada angka 2 =  good, dan angka 3 = excellent. 

Buatlah grafik boxplot untuk variabel kategorik (jenis transmisi dan warna),Ketika menganalisis variabel kategorik, ingatlah bahwa kategori harus memiliki setidaknya 50 iklan. Jika tidak, parameternya tidak akan valid untuk digunakan saat analisis.


```python
# melihat value pada jenis transmisi apakah mempunyai nilai yang lebih dari 50

most_type5_pric['transmission'].value_counts()
```




    automatic    10092
    other          739
    manual         643
    Name: transmission, dtype: int64




```python
# Analisis  5 jenis kendaraan dengan rata-rata harga terbanyak dengan tipe transmisi.

plt.figure(figsize=(10,8))
sns.boxplot(data=most_type5_pric, y='price', x='transmission')

plt.title('Grafik boxplot harga terhadap kondisi')
plt.suptitle("")
plt.show()
```


    
![png](output_163_0.png)
    


Kesimpulan : 

1. Dengan melihat value pada column tipe transmission dimana terdapat categori automatic, other transmissiondan manual dimana semua category mempunyai lebih dari 50 value.


2. Kesimpulan dari boxplot dimana tipe transmisi kendaraan "others transmission" lebih tinggi dibandingkan dengan manual dan automatic. dimana jenis transmisi yang laku pada tipe others transmissi ada pada kisaran 20000 dan diurutan kedua dengan tipe transmisi automatic pada kisaran harga diatas 15000 dan urutan ketiga manual pada kisaran harga dibawah 15000. dengan ini rata-rata kendaraan dengan transmisi other transmission lebih mahal dari kendaraan dengan transmisi automatic dan manual, sehingga jenis transmisi berpengaruh besar terhadap harga.


```python
# melihat value pada warna pada kendaraan apakah mempunyai nilai yang lebih dari 50

most_type5_pric['paint_color'].value_counts()
```




    white     3005
    others    2342
    black     1456
    red       1087
    silver    1046
    grey       936
    blue       843
    custom     215
    green      214
    brown      196
    yellow      79
    orange      42
    purple      13
    Name: paint_color, dtype: int64




```python
# Filter column paint_color yang datanya kurang dari 50 

paint_filter  =  most_type5_pric[most_type5_pric.paint_color.isin(['white','others','black','red','silver','grey','blue','custom','green','brown','yellow'])]
paint_filter['paint_color'].value_counts()
```




    white     3005
    others    2342
    black     1456
    red       1087
    silver    1046
    grey       936
    blue       843
    custom     215
    green      214
    brown      196
    yellow      79
    Name: paint_color, dtype: int64




```python
# Analisis  5 jenis kendaraan dengan rata-rata harga terbanyak dengan tipe warna_kendaraan.

plt.figure(figsize=(13,9))
sns.boxplot(data=most_type5_pric, y='price', x='paint_color')

plt.title('Grafik boxplot harga terhadap warna cat kendaraan ')
plt.suptitle("")
plt.show()
```


    
![png](output_167_0.png)
    


Kesimpulan : 

1. Dengan melihat value pada column paint_color yang memiliki nilai diatas 50 adalah white, others, black, red, silver, grey, blue, custom, green, brown, yellow.

2. Pada boxplot  diatas rata-rata hampir memiliki harga yang hampir sama akan tetapi jika dilihat dengan detail warna dengan harga tertinggi ada pada warna costum, kedua = black, ketiga = white, keempat = red dan kelima others. jadi dapat disimpulkan warna tidak dapat dijadikan acuan untuk harga.

## Kesimpulan Umum :     
---
1. Pra-pemrosesan
Dimulai dari tahap pra-pemrosesan data, dapat didentifikasi missing value dalam data dan missing value tersebut hilang secara acak (MAR). disini saya menggunakan beberapa metode untuk menangani missing value berdasarkan kasus perkasus.
---
2. Mengatasi Nilai-Nilai yang Hilang (Jika Ada)
*   pada column model_year dan silinder saya menghapus nilai yang hilang dengan metode dropna() 
*   Pada column odometer saya mengganti nilai yang hilang dengan median.
*   Pada column pain_color saya membuat kategori baru yakni others.
---
3. Memperbaiki Tipe Data
*   Pada poin ini saya melakukan perubahan pada tipe data yakni dari float64 menjadi int64, column yang tipe datanya perlu dirubah adalah model_year, cylinders, odometer dan is_4wd. alasan perubahan tipe data ini dikarenakan pada saat melakukan aritmatika lebih baik menggunakan bilangan bulat dari pada pecahan atau desimal.
*   Pada Poin ini saya menambahkan untuk mempermudah anilisi pada data :
a. Hari, minggu, bulan, dan tahun kendaraan saat iklan ditayangkan
b. Usia kendaraan (dalam tahun) ketika iklan ditayangkan
c. Jarak tempuh rata-rata kendaraan per tahun
b. Mengganti nilai string dengan skala numerik pada column condition
---
4. Mempelajari Parameter Inti pada column :
*   Pada tahap ini saya melakukan analisis pada column Harga, Usia, kendaraan ketika iklan ditayangkan, Jarak tempuh, Jumlah silinder, Kondisi dan mendapati data yang anomali  yang sebagian besar terdapat outlier dengan kemiringan puncak kiri ke kanan. 
---
5. Mempelajari Parameter Inti Tanpa Outlier :     
*   Pada tahap ini saya melakukan metode Zscore untuk menentukan range data agar distribusi datanya normal setelah itu memfilter data dengan menghilangkan data yang outlier dan dilanjutkan menganalisa data dengan membuat histogram baru. 
---
6. Masa Berlaku Iklan
*  Dari analisis yang dilakukan saya menemukan bahwa masa pakai iklan biasanya sekitar satu bulan. lalu iklan tercepat yang dihapus sebelum 1 hari, dan iklan terlama yang terdaftar selama 271 hari.
---
7. Pada tahap analisis terakhir jenis kendaraan bus dan hatchback adalah jumlah iklan yang paling banyak ditayangkan. disini saya mengelompokkan dan menganalisis lebih dalam tentang faktor apa saja yang mempengaruhi harga kendaraan. dan dari hasil analisis bahwa variabel- variabel di bawah ini merupakan faktor penting dari harga kendaraan:

*   Condition = kondisi kendaraan 
*   Transmission = Tipe transmisi


```python

```
