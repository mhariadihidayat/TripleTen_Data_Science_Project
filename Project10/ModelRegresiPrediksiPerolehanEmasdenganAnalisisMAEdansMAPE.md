# Deskripsi Project :

Pada Project ini kita akan mempersiapkan prototipe Machine Learning untuk Perusahaan Penambangan Emas Zyfra. Perusahaan ini mengenembangkan solulusi untuk efesiensi untuk industri berat. Model dari Machine Learning yang kita persiapkan dapat memprediksi jumlah emas yang diperoleh dari bijih emas. kita diberi akses dara mengenai ekstraksi bahan mentah emas dan pemurnian emas. Tugas yang diberikan adalah Membuat Model Machine Learning yang dapat mengoptimalkan Produksi dan menghilangkan paramater yang tidak menguntungkan. 

---

Technological Process: 

Bagaimana emas diekstraksi dari bijih? Mari kita lihat tahapan prosesnya. 
Bijih yang ditambang mengalami pemrosesan primer untuk mendapatkan campuran bijih atau bahan kasar, yang merupakan bahan baku untuk Flotation (juga dikenal sebagai proses bahan kasar). Setelah flotasi, material dikirim ke pemurnian dua tahap.

1. Flotation Process : 
Campuran bijih emas dimasukkan ke dalam float bank untuk mendapatkan konsentrat Au yang lebih kasar dan ekor yang lebih kasar (residu produk dengan konsentrasi logam mulia yang rendah).
Stabilitas proses ini dipengaruhi oleh keadaan fisikokimia pulp flotasi yang mudah menguap dan tidak optimal (campuran partikel padat dan cair).

2. Purification Process :
Konsentrat yang lebih kasar mengalami dua tahap pemurnian. Setelah pemurnian, kami memiliki konsentrat terakhir dan ekor baru.

---

Deskripsi Data :
1. Technological process :

* Rougher feed — raw material
* Rougher additions (or reagent additions) — flotation reagents: Xanthate, Sulphate, Depressant
* Xanthate — promoter or flotation activator;
* Sulphate — sodium sulphide for this particular process;
* Depressant — sodium silicate.
* Rougher process — flotation
* Rougher tails — product residues
* Float banks — flotation unit
* Cleaner process — purification
* Rougher Au — rougher gold concentrate
* Final Au — final gold concentrate

2. Parameters of stages: 
* air amount — volume of air
* fluid levels
* feed size — feed particle size
* feed rate

---

Tahapan proses teknologi untuk ekstraksi emas dari bijih ditampilkan : 



```python
from IPython.display import Image
Image(url= 'https://pictures.s3.yandex.net/resources/ore_1591699963.jpg',width = 500, height = 200 )
```




<img src="https://pictures.s3.yandex.net/resources/ore_1591699963.jpg" width="500" height="200"/>



Objektif Project : 

1. Simulasikan proses pemulihan emas dari bijih emas

2. Mengembangkan model untuk memprediksi jumlah emas yang diperoleh dan mengoptimalkan produksi emas

3. Menghitung metrik sMAPE value. 



## Prepare the data

# Memuat Libary yang dibutuhkan untuk pemrosesan data


```python
# import pandas and numpy untuk proses dan manipulasi data
import pandas as pd
import numpy as np 

# import timeit untuk perhitungan waktu 
import timeit

# Import seaborn untuk statistika data visualisasi
import seaborn as sns

# import matplotlib untuk data visualisasi
import matplotlib.pyplot as plt 
%matplotlib inline
plt.rcParams.update({'figure.max_open_warning': 0})

# import train_test_split untuk membagi data
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler 
pd.options.mode.chained_assignment = None #menghilangkan notif CopyWarning

# import modul machine learning regressi dari library sklearn
from sklearn.dummy import DummyRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 

# import sanity check untuk memeriksa fungsi terhadap model
from sklearn.metrics import *

# import warnings untuk menghapus peringatan saat dataset di manipulasi
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
```

# Memuat Data dari csv agar dapat dijalankan dengan pandas untuk menjadi DataFrame


```python
gold_recovery_train = pd.read_csv('/datasets/gold_recovery_train.csv')
gold_recovery_test  = pd.read_csv('/datasets/gold_recovery_test.csv')
gold_recovery_full   = pd.read_csv('/datasets/gold_recovery_full.csv')
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
    display(df.head())
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

### Mempelajari informasi umum pada keseluruhan dataset  



```python
print('Informasi Umum pada Dataset gold_recovery_train')
get_info(gold_recovery_train)
```

    Informasi Umum pada Dataset gold_recovery_train
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
      <th>date</th>
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-15 00:00:00</td>
      <td>6.055403</td>
      <td>9.889648</td>
      <td>5.507324</td>
      <td>42.192020</td>
      <td>70.541216</td>
      <td>10.411962</td>
      <td>0.895447</td>
      <td>16.904297</td>
      <td>2.143149</td>
      <td>...</td>
      <td>14.016835</td>
      <td>-502.488007</td>
      <td>12.099931</td>
      <td>-504.715942</td>
      <td>9.925633</td>
      <td>-498.310211</td>
      <td>8.079666</td>
      <td>-500.470978</td>
      <td>14.151341</td>
      <td>-605.841980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-15 01:00:00</td>
      <td>6.029369</td>
      <td>9.968944</td>
      <td>5.257781</td>
      <td>42.701629</td>
      <td>69.266198</td>
      <td>10.462676</td>
      <td>0.927452</td>
      <td>16.634514</td>
      <td>2.224930</td>
      <td>...</td>
      <td>13.992281</td>
      <td>-505.503262</td>
      <td>11.950531</td>
      <td>-501.331529</td>
      <td>10.039245</td>
      <td>-500.169983</td>
      <td>7.984757</td>
      <td>-500.582168</td>
      <td>13.998353</td>
      <td>-599.787184</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-15 02:00:00</td>
      <td>6.055926</td>
      <td>10.213995</td>
      <td>5.383759</td>
      <td>42.657501</td>
      <td>68.116445</td>
      <td>10.507046</td>
      <td>0.953716</td>
      <td>16.208849</td>
      <td>2.257889</td>
      <td>...</td>
      <td>14.015015</td>
      <td>-502.520901</td>
      <td>11.912783</td>
      <td>-501.133383</td>
      <td>10.070913</td>
      <td>-500.129135</td>
      <td>8.013877</td>
      <td>-500.517572</td>
      <td>14.028663</td>
      <td>-601.427363</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-15 03:00:00</td>
      <td>6.047977</td>
      <td>9.977019</td>
      <td>4.858634</td>
      <td>42.689819</td>
      <td>68.347543</td>
      <td>10.422762</td>
      <td>0.883763</td>
      <td>16.532835</td>
      <td>2.146849</td>
      <td>...</td>
      <td>14.036510</td>
      <td>-500.857308</td>
      <td>11.999550</td>
      <td>-501.193686</td>
      <td>9.970366</td>
      <td>-499.201640</td>
      <td>7.977324</td>
      <td>-500.255908</td>
      <td>14.005551</td>
      <td>-599.996129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-15 04:00:00</td>
      <td>6.148599</td>
      <td>10.142511</td>
      <td>4.939416</td>
      <td>42.774141</td>
      <td>66.927016</td>
      <td>10.360302</td>
      <td>0.792826</td>
      <td>16.525686</td>
      <td>2.055292</td>
      <td>...</td>
      <td>14.027298</td>
      <td>-499.838632</td>
      <td>11.953070</td>
      <td>-501.053894</td>
      <td>9.925709</td>
      <td>-501.686727</td>
      <td>7.894242</td>
      <td>-500.356035</td>
      <td>13.996647</td>
      <td>-601.496691</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 87 columns</p>
</div>


    ----------------------------------------------------------------------------------------------------
    Info:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16860 entries, 0 to 16859
    Data columns (total 87 columns):
     #   Column                                              Non-Null Count  Dtype  
    ---  ------                                              --------------  -----  
     0   date                                                16860 non-null  object 
     1   final.output.concentrate_ag                         16788 non-null  float64
     2   final.output.concentrate_pb                         16788 non-null  float64
     3   final.output.concentrate_sol                        16490 non-null  float64
     4   final.output.concentrate_au                         16789 non-null  float64
     5   final.output.recovery                               15339 non-null  float64
     6   final.output.tail_ag                                16794 non-null  float64
     7   final.output.tail_pb                                16677 non-null  float64
     8   final.output.tail_sol                               16715 non-null  float64
     9   final.output.tail_au                                16794 non-null  float64
     10  primary_cleaner.input.sulfate                       15553 non-null  float64
     11  primary_cleaner.input.depressant                    15598 non-null  float64
     12  primary_cleaner.input.feed_size                     16860 non-null  float64
     13  primary_cleaner.input.xanthate                      15875 non-null  float64
     14  primary_cleaner.output.concentrate_ag               16778 non-null  float64
     15  primary_cleaner.output.concentrate_pb               16502 non-null  float64
     16  primary_cleaner.output.concentrate_sol              16224 non-null  float64
     17  primary_cleaner.output.concentrate_au               16778 non-null  float64
     18  primary_cleaner.output.tail_ag                      16777 non-null  float64
     19  primary_cleaner.output.tail_pb                      16761 non-null  float64
     20  primary_cleaner.output.tail_sol                     16579 non-null  float64
     21  primary_cleaner.output.tail_au                      16777 non-null  float64
     22  primary_cleaner.state.floatbank8_a_air              16820 non-null  float64
     23  primary_cleaner.state.floatbank8_a_level            16827 non-null  float64
     24  primary_cleaner.state.floatbank8_b_air              16820 non-null  float64
     25  primary_cleaner.state.floatbank8_b_level            16833 non-null  float64
     26  primary_cleaner.state.floatbank8_c_air              16822 non-null  float64
     27  primary_cleaner.state.floatbank8_c_level            16833 non-null  float64
     28  primary_cleaner.state.floatbank8_d_air              16821 non-null  float64
     29  primary_cleaner.state.floatbank8_d_level            16833 non-null  float64
     30  rougher.calculation.sulfate_to_au_concentrate       16833 non-null  float64
     31  rougher.calculation.floatbank10_sulfate_to_au_feed  16833 non-null  float64
     32  rougher.calculation.floatbank11_sulfate_to_au_feed  16833 non-null  float64
     33  rougher.calculation.au_pb_ratio                     15618 non-null  float64
     34  rougher.input.feed_ag                               16778 non-null  float64
     35  rougher.input.feed_pb                               16632 non-null  float64
     36  rougher.input.feed_rate                             16347 non-null  float64
     37  rougher.input.feed_size                             16443 non-null  float64
     38  rougher.input.feed_sol                              16568 non-null  float64
     39  rougher.input.feed_au                               16777 non-null  float64
     40  rougher.input.floatbank10_sulfate                   15816 non-null  float64
     41  rougher.input.floatbank10_xanthate                  16514 non-null  float64
     42  rougher.input.floatbank11_sulfate                   16237 non-null  float64
     43  rougher.input.floatbank11_xanthate                  14956 non-null  float64
     44  rougher.output.concentrate_ag                       16778 non-null  float64
     45  rougher.output.concentrate_pb                       16778 non-null  float64
     46  rougher.output.concentrate_sol                      16698 non-null  float64
     47  rougher.output.concentrate_au                       16778 non-null  float64
     48  rougher.output.recovery                             14287 non-null  float64
     49  rougher.output.tail_ag                              14610 non-null  float64
     50  rougher.output.tail_pb                              16778 non-null  float64
     51  rougher.output.tail_sol                             14611 non-null  float64
     52  rougher.output.tail_au                              14611 non-null  float64
     53  rougher.state.floatbank10_a_air                     16807 non-null  float64
     54  rougher.state.floatbank10_a_level                   16807 non-null  float64
     55  rougher.state.floatbank10_b_air                     16807 non-null  float64
     56  rougher.state.floatbank10_b_level                   16807 non-null  float64
     57  rougher.state.floatbank10_c_air                     16807 non-null  float64
     58  rougher.state.floatbank10_c_level                   16814 non-null  float64
     59  rougher.state.floatbank10_d_air                     16802 non-null  float64
     60  rougher.state.floatbank10_d_level                   16809 non-null  float64
     61  rougher.state.floatbank10_e_air                     16257 non-null  float64
     62  rougher.state.floatbank10_e_level                   16809 non-null  float64
     63  rougher.state.floatbank10_f_air                     16802 non-null  float64
     64  rougher.state.floatbank10_f_level                   16802 non-null  float64
     65  secondary_cleaner.output.tail_ag                    16776 non-null  float64
     66  secondary_cleaner.output.tail_pb                    16764 non-null  float64
     67  secondary_cleaner.output.tail_sol                   14874 non-null  float64
     68  secondary_cleaner.output.tail_au                    16778 non-null  float64
     69  secondary_cleaner.state.floatbank2_a_air            16497 non-null  float64
     70  secondary_cleaner.state.floatbank2_a_level          16751 non-null  float64
     71  secondary_cleaner.state.floatbank2_b_air            16705 non-null  float64
     72  secondary_cleaner.state.floatbank2_b_level          16748 non-null  float64
     73  secondary_cleaner.state.floatbank3_a_air            16763 non-null  float64
     74  secondary_cleaner.state.floatbank3_a_level          16747 non-null  float64
     75  secondary_cleaner.state.floatbank3_b_air            16752 non-null  float64
     76  secondary_cleaner.state.floatbank3_b_level          16750 non-null  float64
     77  secondary_cleaner.state.floatbank4_a_air            16731 non-null  float64
     78  secondary_cleaner.state.floatbank4_a_level          16747 non-null  float64
     79  secondary_cleaner.state.floatbank4_b_air            16768 non-null  float64
     80  secondary_cleaner.state.floatbank4_b_level          16767 non-null  float64
     81  secondary_cleaner.state.floatbank5_a_air            16775 non-null  float64
     82  secondary_cleaner.state.floatbank5_a_level          16775 non-null  float64
     83  secondary_cleaner.state.floatbank5_b_air            16775 non-null  float64
     84  secondary_cleaner.state.floatbank5_b_level          16776 non-null  float64
     85  secondary_cleaner.state.floatbank6_a_air            16757 non-null  float64
     86  secondary_cleaner.state.floatbank6_a_level          16775 non-null  float64
    dtypes: float64(86), object(1)
    memory usage: 11.2+ MB



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
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>primary_cleaner.input.sulfate</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16788.000000</td>
      <td>16788.000000</td>
      <td>16490.000000</td>
      <td>16789.000000</td>
      <td>15339.000000</td>
      <td>16794.000000</td>
      <td>16677.000000</td>
      <td>16715.000000</td>
      <td>16794.000000</td>
      <td>15553.000000</td>
      <td>...</td>
      <td>16731.000000</td>
      <td>16747.000000</td>
      <td>16768.000000</td>
      <td>16767.000000</td>
      <td>16775.000000</td>
      <td>16775.000000</td>
      <td>16775.000000</td>
      <td>16776.000000</td>
      <td>16757.000000</td>
      <td>16775.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.716907</td>
      <td>9.113559</td>
      <td>8.301123</td>
      <td>39.467217</td>
      <td>67.213166</td>
      <td>8.757048</td>
      <td>2.360327</td>
      <td>9.303932</td>
      <td>2.687512</td>
      <td>129.479789</td>
      <td>...</td>
      <td>19.101874</td>
      <td>-494.164481</td>
      <td>14.778164</td>
      <td>-476.600082</td>
      <td>15.779488</td>
      <td>-500.230146</td>
      <td>12.377241</td>
      <td>-498.956257</td>
      <td>18.429208</td>
      <td>-521.801826</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.096718</td>
      <td>3.389495</td>
      <td>3.825760</td>
      <td>13.917227</td>
      <td>11.960446</td>
      <td>3.634103</td>
      <td>1.215576</td>
      <td>4.263208</td>
      <td>1.272757</td>
      <td>45.386931</td>
      <td>...</td>
      <td>6.883163</td>
      <td>84.803334</td>
      <td>5.999149</td>
      <td>89.381172</td>
      <td>6.834703</td>
      <td>76.983542</td>
      <td>6.219989</td>
      <td>82.146207</td>
      <td>6.958294</td>
      <td>77.170888</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000003</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-799.920713</td>
      <td>0.000000</td>
      <td>-800.021781</td>
      <td>-0.423260</td>
      <td>-799.741097</td>
      <td>0.427084</td>
      <td>-800.258209</td>
      <td>0.024270</td>
      <td>-810.473526</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.971262</td>
      <td>8.825748</td>
      <td>6.939185</td>
      <td>42.055722</td>
      <td>62.625685</td>
      <td>7.610544</td>
      <td>1.641604</td>
      <td>7.870275</td>
      <td>2.172953</td>
      <td>103.064021</td>
      <td>...</td>
      <td>14.508299</td>
      <td>-500.837689</td>
      <td>10.741388</td>
      <td>-500.269182</td>
      <td>10.977713</td>
      <td>-500.530594</td>
      <td>8.925586</td>
      <td>-500.147603</td>
      <td>13.977626</td>
      <td>-501.080595</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.869346</td>
      <td>10.065316</td>
      <td>8.557228</td>
      <td>44.498874</td>
      <td>67.644601</td>
      <td>9.220393</td>
      <td>2.453690</td>
      <td>10.021968</td>
      <td>2.781132</td>
      <td>131.783108</td>
      <td>...</td>
      <td>19.986958</td>
      <td>-499.778379</td>
      <td>14.943933</td>
      <td>-499.593286</td>
      <td>15.998340</td>
      <td>-499.784231</td>
      <td>11.092839</td>
      <td>-499.933330</td>
      <td>18.034960</td>
      <td>-500.109898</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.821176</td>
      <td>11.054809</td>
      <td>10.289741</td>
      <td>45.976222</td>
      <td>72.824595</td>
      <td>10.971110</td>
      <td>3.192404</td>
      <td>11.648573</td>
      <td>3.416936</td>
      <td>159.539839</td>
      <td>...</td>
      <td>24.983961</td>
      <td>-494.648754</td>
      <td>20.023751</td>
      <td>-400.137948</td>
      <td>20.000701</td>
      <td>-496.531781</td>
      <td>15.979467</td>
      <td>-498.418000</td>
      <td>24.984992</td>
      <td>-499.565540</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.001945</td>
      <td>17.031899</td>
      <td>18.124851</td>
      <td>53.611374</td>
      <td>100.000000</td>
      <td>19.552149</td>
      <td>6.086532</td>
      <td>22.317730</td>
      <td>9.789625</td>
      <td>251.999948</td>
      <td>...</td>
      <td>60.000000</td>
      <td>-127.692333</td>
      <td>28.003828</td>
      <td>-71.472472</td>
      <td>63.116298</td>
      <td>-275.073125</td>
      <td>39.846228</td>
      <td>-120.190931</td>
      <td>54.876806</td>
      <td>-39.784927</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 86 columns</p>
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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16860</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>16860</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2018-05-19 07:59:59</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Columns dengan nilai yang hilang:
    Column final.output.concentrate_ag dengan 0.4270% persentasi nilai yang hilang , dan 72 nilai yang hilang
    Column final.output.concentrate_pb dengan 0.4270% persentasi nilai yang hilang , dan 72 nilai yang hilang
    Column final.output.concentrate_sol dengan 2.1945% persentasi nilai yang hilang , dan 370 nilai yang hilang
    Column final.output.concentrate_au dengan 0.4211% persentasi nilai yang hilang , dan 71 nilai yang hilang
    Column final.output.recovery dengan 9.0214% persentasi nilai yang hilang , dan 1521 nilai yang hilang
    Column final.output.tail_ag dengan 0.3915% persentasi nilai yang hilang , dan 66 nilai yang hilang
    Column final.output.tail_pb dengan 1.0854% persentasi nilai yang hilang , dan 183 nilai yang hilang
    Column final.output.tail_sol dengan 0.8600% persentasi nilai yang hilang , dan 145 nilai yang hilang
    Column final.output.tail_au dengan 0.3915% persentasi nilai yang hilang , dan 66 nilai yang hilang
    Column primary_cleaner.input.sulfate dengan 7.7521% persentasi nilai yang hilang , dan 1307 nilai yang hilang
    Column primary_cleaner.input.depressant dengan 7.4852% persentasi nilai yang hilang , dan 1262 nilai yang hilang
    Column primary_cleaner.input.xanthate dengan 5.8422% persentasi nilai yang hilang , dan 985 nilai yang hilang
    Column primary_cleaner.output.concentrate_ag dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column primary_cleaner.output.concentrate_pb dengan 2.1234% persentasi nilai yang hilang , dan 358 nilai yang hilang
    Column primary_cleaner.output.concentrate_sol dengan 3.7722% persentasi nilai yang hilang , dan 636 nilai yang hilang
    Column primary_cleaner.output.concentrate_au dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column primary_cleaner.output.tail_ag dengan 0.4923% persentasi nilai yang hilang , dan 83 nilai yang hilang
    Column primary_cleaner.output.tail_pb dengan 0.5872% persentasi nilai yang hilang , dan 99 nilai yang hilang
    Column primary_cleaner.output.tail_sol dengan 1.6667% persentasi nilai yang hilang , dan 281 nilai yang hilang
    Column primary_cleaner.output.tail_au dengan 0.4923% persentasi nilai yang hilang , dan 83 nilai yang hilang
    Column primary_cleaner.state.floatbank8_a_air dengan 0.2372% persentasi nilai yang hilang , dan 40 nilai yang hilang
    Column primary_cleaner.state.floatbank8_a_level dengan 0.1957% persentasi nilai yang hilang , dan 33 nilai yang hilang
    Column primary_cleaner.state.floatbank8_b_air dengan 0.2372% persentasi nilai yang hilang , dan 40 nilai yang hilang
    Column primary_cleaner.state.floatbank8_b_level dengan 0.1601% persentasi nilai yang hilang , dan 27 nilai yang hilang
    Column primary_cleaner.state.floatbank8_c_air dengan 0.2254% persentasi nilai yang hilang , dan 38 nilai yang hilang
    Column primary_cleaner.state.floatbank8_c_level dengan 0.1601% persentasi nilai yang hilang , dan 27 nilai yang hilang
    Column primary_cleaner.state.floatbank8_d_air dengan 0.2313% persentasi nilai yang hilang , dan 39 nilai yang hilang
    Column primary_cleaner.state.floatbank8_d_level dengan 0.1601% persentasi nilai yang hilang , dan 27 nilai yang hilang
    Column rougher.calculation.sulfate_to_au_concentrate dengan 0.1601% persentasi nilai yang hilang , dan 27 nilai yang hilang
    Column rougher.calculation.floatbank10_sulfate_to_au_feed dengan 0.1601% persentasi nilai yang hilang , dan 27 nilai yang hilang
    Column rougher.calculation.floatbank11_sulfate_to_au_feed dengan 0.1601% persentasi nilai yang hilang , dan 27 nilai yang hilang
    Column rougher.calculation.au_pb_ratio dengan 7.3665% persentasi nilai yang hilang , dan 1242 nilai yang hilang
    Column rougher.input.feed_ag dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column rougher.input.feed_pb dengan 1.3523% persentasi nilai yang hilang , dan 228 nilai yang hilang
    Column rougher.input.feed_rate dengan 3.0427% persentasi nilai yang hilang , dan 513 nilai yang hilang
    Column rougher.input.feed_size dengan 2.4733% persentasi nilai yang hilang , dan 417 nilai yang hilang
    Column rougher.input.feed_sol dengan 1.7319% persentasi nilai yang hilang , dan 292 nilai yang hilang
    Column rougher.input.feed_au dengan 0.4923% persentasi nilai yang hilang , dan 83 nilai yang hilang
    Column rougher.input.floatbank10_sulfate dengan 6.1922% persentasi nilai yang hilang , dan 1044 nilai yang hilang
    Column rougher.input.floatbank10_xanthate dengan 2.0522% persentasi nilai yang hilang , dan 346 nilai yang hilang
    Column rougher.input.floatbank11_sulfate dengan 3.6951% persentasi nilai yang hilang , dan 623 nilai yang hilang
    Column rougher.input.floatbank11_xanthate dengan 11.2930% persentasi nilai yang hilang , dan 1904 nilai yang hilang
    Column rougher.output.concentrate_ag dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column rougher.output.concentrate_pb dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column rougher.output.concentrate_sol dengan 0.9609% persentasi nilai yang hilang , dan 162 nilai yang hilang
    Column rougher.output.concentrate_au dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column rougher.output.recovery dengan 15.2610% persentasi nilai yang hilang , dan 2573 nilai yang hilang
    Column rougher.output.tail_ag dengan 13.3452% persentasi nilai yang hilang , dan 2250 nilai yang hilang
    Column rougher.output.tail_pb dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column rougher.output.tail_sol dengan 13.3393% persentasi nilai yang hilang , dan 2249 nilai yang hilang
    Column rougher.output.tail_au dengan 13.3393% persentasi nilai yang hilang , dan 2249 nilai yang hilang
    Column rougher.state.floatbank10_a_air dengan 0.3144% persentasi nilai yang hilang , dan 53 nilai yang hilang
    Column rougher.state.floatbank10_a_level dengan 0.3144% persentasi nilai yang hilang , dan 53 nilai yang hilang
    Column rougher.state.floatbank10_b_air dengan 0.3144% persentasi nilai yang hilang , dan 53 nilai yang hilang
    Column rougher.state.floatbank10_b_level dengan 0.3144% persentasi nilai yang hilang , dan 53 nilai yang hilang
    Column rougher.state.floatbank10_c_air dengan 0.3144% persentasi nilai yang hilang , dan 53 nilai yang hilang
    Column rougher.state.floatbank10_c_level dengan 0.2728% persentasi nilai yang hilang , dan 46 nilai yang hilang
    Column rougher.state.floatbank10_d_air dengan 0.3440% persentasi nilai yang hilang , dan 58 nilai yang hilang
    Column rougher.state.floatbank10_d_level dengan 0.3025% persentasi nilai yang hilang , dan 51 nilai yang hilang
    Column rougher.state.floatbank10_e_air dengan 3.5765% persentasi nilai yang hilang , dan 603 nilai yang hilang
    Column rougher.state.floatbank10_e_level dengan 0.3025% persentasi nilai yang hilang , dan 51 nilai yang hilang
    Column rougher.state.floatbank10_f_air dengan 0.3440% persentasi nilai yang hilang , dan 58 nilai yang hilang
    Column rougher.state.floatbank10_f_level dengan 0.3440% persentasi nilai yang hilang , dan 58 nilai yang hilang
    Column secondary_cleaner.output.tail_ag dengan 0.4982% persentasi nilai yang hilang , dan 84 nilai yang hilang
    Column secondary_cleaner.output.tail_pb dengan 0.5694% persentasi nilai yang hilang , dan 96 nilai yang hilang
    Column secondary_cleaner.output.tail_sol dengan 11.7794% persentasi nilai yang hilang , dan 1986 nilai yang hilang
    Column secondary_cleaner.output.tail_au dengan 0.4864% persentasi nilai yang hilang , dan 82 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_a_air dengan 2.1530% persentasi nilai yang hilang , dan 363 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_a_level dengan 0.6465% persentasi nilai yang hilang , dan 109 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_b_air dengan 0.9193% persentasi nilai yang hilang , dan 155 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_b_level dengan 0.6643% persentasi nilai yang hilang , dan 112 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_a_air dengan 0.5753% persentasi nilai yang hilang , dan 97 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_a_level dengan 0.6702% persentasi nilai yang hilang , dan 113 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_b_air dengan 0.6406% persentasi nilai yang hilang , dan 108 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_b_level dengan 0.6524% persentasi nilai yang hilang , dan 110 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_a_air dengan 0.7651% persentasi nilai yang hilang , dan 129 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_a_level dengan 0.6702% persentasi nilai yang hilang , dan 113 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_b_air dengan 0.5457% persentasi nilai yang hilang , dan 92 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_b_level dengan 0.5516% persentasi nilai yang hilang , dan 93 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_a_air dengan 0.5042% persentasi nilai yang hilang , dan 85 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_a_level dengan 0.5042% persentasi nilai yang hilang , dan 85 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_b_air dengan 0.5042% persentasi nilai yang hilang , dan 85 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_b_level dengan 0.4982% persentasi nilai yang hilang , dan 84 nilai yang hilang
    Column secondary_cleaner.state.floatbank6_a_air dengan 0.6109% persentasi nilai yang hilang , dan 103 nilai yang hilang
    Column secondary_cleaner.state.floatbank6_a_level dengan 0.5042% persentasi nilai yang hilang , dan 85 nilai yang hilang
    [1mTerdapat 85 columns dengan nilai NA.[0m



    None


    ----------------------------------------------------------------------------------------------------
    Shape:
    (16860, 87)
    ----------------------------------------------------------------------------------------------------
    Duplicated:
    [1mKita mempunyai 0 baris yang terduplikasi.
    [0m
    



```python
print('Informasi Umum pada Dataset gold_recovery_test')
get_info(gold_recovery_test)
```

    Informasi Umum pada Dataset gold_recovery_test
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
      <th>date</th>
      <th>primary_cleaner.input.sulfate</th>
      <th>primary_cleaner.input.depressant</th>
      <th>primary_cleaner.input.feed_size</th>
      <th>primary_cleaner.input.xanthate</th>
      <th>primary_cleaner.state.floatbank8_a_air</th>
      <th>primary_cleaner.state.floatbank8_a_level</th>
      <th>primary_cleaner.state.floatbank8_b_air</th>
      <th>primary_cleaner.state.floatbank8_b_level</th>
      <th>primary_cleaner.state.floatbank8_c_air</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-09-01 00:59:59</td>
      <td>210.800909</td>
      <td>14.993118</td>
      <td>8.080000</td>
      <td>1.005021</td>
      <td>1398.981301</td>
      <td>-500.225577</td>
      <td>1399.144926</td>
      <td>-499.919735</td>
      <td>1400.102998</td>
      <td>...</td>
      <td>12.023554</td>
      <td>-497.795834</td>
      <td>8.016656</td>
      <td>-501.289139</td>
      <td>7.946562</td>
      <td>-432.317850</td>
      <td>4.872511</td>
      <td>-500.037437</td>
      <td>26.705889</td>
      <td>-499.709414</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-09-01 01:59:59</td>
      <td>215.392455</td>
      <td>14.987471</td>
      <td>8.080000</td>
      <td>0.990469</td>
      <td>1398.777912</td>
      <td>-500.057435</td>
      <td>1398.055362</td>
      <td>-499.778182</td>
      <td>1396.151033</td>
      <td>...</td>
      <td>12.058140</td>
      <td>-498.695773</td>
      <td>8.130979</td>
      <td>-499.634209</td>
      <td>7.958270</td>
      <td>-525.839648</td>
      <td>4.878850</td>
      <td>-500.162375</td>
      <td>25.019940</td>
      <td>-499.819438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-09-01 02:59:59</td>
      <td>215.259946</td>
      <td>12.884934</td>
      <td>7.786667</td>
      <td>0.996043</td>
      <td>1398.493666</td>
      <td>-500.868360</td>
      <td>1398.860436</td>
      <td>-499.764529</td>
      <td>1398.075709</td>
      <td>...</td>
      <td>11.962366</td>
      <td>-498.767484</td>
      <td>8.096893</td>
      <td>-500.827423</td>
      <td>8.071056</td>
      <td>-500.801673</td>
      <td>4.905125</td>
      <td>-499.828510</td>
      <td>24.994862</td>
      <td>-500.622559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-09-01 03:59:59</td>
      <td>215.336236</td>
      <td>12.006805</td>
      <td>7.640000</td>
      <td>0.863514</td>
      <td>1399.618111</td>
      <td>-498.863574</td>
      <td>1397.440120</td>
      <td>-499.211024</td>
      <td>1400.129303</td>
      <td>...</td>
      <td>12.033091</td>
      <td>-498.350935</td>
      <td>8.074946</td>
      <td>-499.474407</td>
      <td>7.897085</td>
      <td>-500.868509</td>
      <td>4.931400</td>
      <td>-499.963623</td>
      <td>24.948919</td>
      <td>-498.709987</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-09-01 04:59:59</td>
      <td>199.099327</td>
      <td>10.682530</td>
      <td>7.530000</td>
      <td>0.805575</td>
      <td>1401.268123</td>
      <td>-500.808305</td>
      <td>1398.128818</td>
      <td>-499.504543</td>
      <td>1402.172226</td>
      <td>...</td>
      <td>12.025367</td>
      <td>-500.786497</td>
      <td>8.054678</td>
      <td>-500.397500</td>
      <td>8.107890</td>
      <td>-509.526725</td>
      <td>4.957674</td>
      <td>-500.360026</td>
      <td>25.003331</td>
      <td>-500.856333</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>


    ----------------------------------------------------------------------------------------------------
    Info:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5856 entries, 0 to 5855
    Data columns (total 53 columns):
     #   Column                                      Non-Null Count  Dtype  
    ---  ------                                      --------------  -----  
     0   date                                        5856 non-null   object 
     1   primary_cleaner.input.sulfate               5554 non-null   float64
     2   primary_cleaner.input.depressant            5572 non-null   float64
     3   primary_cleaner.input.feed_size             5856 non-null   float64
     4   primary_cleaner.input.xanthate              5690 non-null   float64
     5   primary_cleaner.state.floatbank8_a_air      5840 non-null   float64
     6   primary_cleaner.state.floatbank8_a_level    5840 non-null   float64
     7   primary_cleaner.state.floatbank8_b_air      5840 non-null   float64
     8   primary_cleaner.state.floatbank8_b_level    5840 non-null   float64
     9   primary_cleaner.state.floatbank8_c_air      5840 non-null   float64
     10  primary_cleaner.state.floatbank8_c_level    5840 non-null   float64
     11  primary_cleaner.state.floatbank8_d_air      5840 non-null   float64
     12  primary_cleaner.state.floatbank8_d_level    5840 non-null   float64
     13  rougher.input.feed_ag                       5840 non-null   float64
     14  rougher.input.feed_pb                       5840 non-null   float64
     15  rougher.input.feed_rate                     5816 non-null   float64
     16  rougher.input.feed_size                     5834 non-null   float64
     17  rougher.input.feed_sol                      5789 non-null   float64
     18  rougher.input.feed_au                       5840 non-null   float64
     19  rougher.input.floatbank10_sulfate           5599 non-null   float64
     20  rougher.input.floatbank10_xanthate          5733 non-null   float64
     21  rougher.input.floatbank11_sulfate           5801 non-null   float64
     22  rougher.input.floatbank11_xanthate          5503 non-null   float64
     23  rougher.state.floatbank10_a_air             5839 non-null   float64
     24  rougher.state.floatbank10_a_level           5840 non-null   float64
     25  rougher.state.floatbank10_b_air             5839 non-null   float64
     26  rougher.state.floatbank10_b_level           5840 non-null   float64
     27  rougher.state.floatbank10_c_air             5839 non-null   float64
     28  rougher.state.floatbank10_c_level           5840 non-null   float64
     29  rougher.state.floatbank10_d_air             5839 non-null   float64
     30  rougher.state.floatbank10_d_level           5840 non-null   float64
     31  rougher.state.floatbank10_e_air             5839 non-null   float64
     32  rougher.state.floatbank10_e_level           5840 non-null   float64
     33  rougher.state.floatbank10_f_air             5839 non-null   float64
     34  rougher.state.floatbank10_f_level           5840 non-null   float64
     35  secondary_cleaner.state.floatbank2_a_air    5836 non-null   float64
     36  secondary_cleaner.state.floatbank2_a_level  5840 non-null   float64
     37  secondary_cleaner.state.floatbank2_b_air    5833 non-null   float64
     38  secondary_cleaner.state.floatbank2_b_level  5840 non-null   float64
     39  secondary_cleaner.state.floatbank3_a_air    5822 non-null   float64
     40  secondary_cleaner.state.floatbank3_a_level  5840 non-null   float64
     41  secondary_cleaner.state.floatbank3_b_air    5840 non-null   float64
     42  secondary_cleaner.state.floatbank3_b_level  5840 non-null   float64
     43  secondary_cleaner.state.floatbank4_a_air    5840 non-null   float64
     44  secondary_cleaner.state.floatbank4_a_level  5840 non-null   float64
     45  secondary_cleaner.state.floatbank4_b_air    5840 non-null   float64
     46  secondary_cleaner.state.floatbank4_b_level  5840 non-null   float64
     47  secondary_cleaner.state.floatbank5_a_air    5840 non-null   float64
     48  secondary_cleaner.state.floatbank5_a_level  5840 non-null   float64
     49  secondary_cleaner.state.floatbank5_b_air    5840 non-null   float64
     50  secondary_cleaner.state.floatbank5_b_level  5840 non-null   float64
     51  secondary_cleaner.state.floatbank6_a_air    5840 non-null   float64
     52  secondary_cleaner.state.floatbank6_a_level  5840 non-null   float64
    dtypes: float64(52), object(1)
    memory usage: 2.4+ MB



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
      <th>primary_cleaner.input.sulfate</th>
      <th>primary_cleaner.input.depressant</th>
      <th>primary_cleaner.input.feed_size</th>
      <th>primary_cleaner.input.xanthate</th>
      <th>primary_cleaner.state.floatbank8_a_air</th>
      <th>primary_cleaner.state.floatbank8_a_level</th>
      <th>primary_cleaner.state.floatbank8_b_air</th>
      <th>primary_cleaner.state.floatbank8_b_level</th>
      <th>primary_cleaner.state.floatbank8_c_air</th>
      <th>primary_cleaner.state.floatbank8_c_level</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5554.000000</td>
      <td>5572.000000</td>
      <td>5856.000000</td>
      <td>5690.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>...</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
      <td>5840.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>170.515243</td>
      <td>8.482873</td>
      <td>7.264651</td>
      <td>1.321420</td>
      <td>1481.990241</td>
      <td>-509.057796</td>
      <td>1486.908670</td>
      <td>-511.743956</td>
      <td>1468.495216</td>
      <td>-509.741212</td>
      <td>...</td>
      <td>15.636031</td>
      <td>-516.266074</td>
      <td>13.145702</td>
      <td>-476.338907</td>
      <td>12.308967</td>
      <td>-512.208126</td>
      <td>9.470986</td>
      <td>-505.017827</td>
      <td>16.678722</td>
      <td>-512.351694</td>
    </tr>
    <tr>
      <th>std</th>
      <td>49.608602</td>
      <td>3.353105</td>
      <td>0.611526</td>
      <td>0.693246</td>
      <td>310.453166</td>
      <td>61.339256</td>
      <td>313.224286</td>
      <td>67.139074</td>
      <td>309.980748</td>
      <td>62.671873</td>
      <td>...</td>
      <td>4.660835</td>
      <td>62.756748</td>
      <td>4.304086</td>
      <td>105.549424</td>
      <td>3.762827</td>
      <td>58.864651</td>
      <td>3.312471</td>
      <td>68.785898</td>
      <td>5.404514</td>
      <td>69.919839</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000103</td>
      <td>0.000031</td>
      <td>5.650000</td>
      <td>0.000003</td>
      <td>0.000000</td>
      <td>-799.773788</td>
      <td>0.000000</td>
      <td>-800.029078</td>
      <td>0.000000</td>
      <td>-799.995127</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-799.798523</td>
      <td>0.000000</td>
      <td>-800.836914</td>
      <td>-0.223393</td>
      <td>-799.661076</td>
      <td>0.528083</td>
      <td>-800.220337</td>
      <td>-0.079426</td>
      <td>-809.859706</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>143.340022</td>
      <td>6.411500</td>
      <td>6.885625</td>
      <td>0.888769</td>
      <td>1497.190681</td>
      <td>-500.455211</td>
      <td>1497.150234</td>
      <td>-500.936639</td>
      <td>1437.050321</td>
      <td>-501.300441</td>
      <td>...</td>
      <td>12.057838</td>
      <td>-501.054741</td>
      <td>11.880119</td>
      <td>-500.419113</td>
      <td>10.123459</td>
      <td>-500.879383</td>
      <td>7.991208</td>
      <td>-500.223089</td>
      <td>13.012422</td>
      <td>-500.833821</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>176.103893</td>
      <td>8.023252</td>
      <td>7.259333</td>
      <td>1.183362</td>
      <td>1554.659783</td>
      <td>-499.997402</td>
      <td>1553.268084</td>
      <td>-500.066588</td>
      <td>1546.160672</td>
      <td>-500.079537</td>
      <td>...</td>
      <td>17.001867</td>
      <td>-500.160145</td>
      <td>14.952102</td>
      <td>-499.644328</td>
      <td>12.062877</td>
      <td>-500.047621</td>
      <td>9.980774</td>
      <td>-500.001338</td>
      <td>16.007242</td>
      <td>-500.041085</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>207.240761</td>
      <td>10.017725</td>
      <td>7.650000</td>
      <td>1.763797</td>
      <td>1601.681656</td>
      <td>-499.575313</td>
      <td>1601.784707</td>
      <td>-499.323361</td>
      <td>1600.785573</td>
      <td>-499.009545</td>
      <td>...</td>
      <td>18.030985</td>
      <td>-499.441529</td>
      <td>15.940011</td>
      <td>-401.523664</td>
      <td>15.017881</td>
      <td>-499.297033</td>
      <td>11.992176</td>
      <td>-499.722835</td>
      <td>21.009076</td>
      <td>-499.395621</td>
    </tr>
    <tr>
      <th>max</th>
      <td>274.409626</td>
      <td>40.024582</td>
      <td>15.500000</td>
      <td>5.433169</td>
      <td>2212.432090</td>
      <td>-57.195404</td>
      <td>1975.147923</td>
      <td>-142.527229</td>
      <td>1715.053773</td>
      <td>-150.937035</td>
      <td>...</td>
      <td>30.051797</td>
      <td>-401.565212</td>
      <td>31.269706</td>
      <td>-6.506986</td>
      <td>25.258848</td>
      <td>-244.483566</td>
      <td>14.090194</td>
      <td>-126.463446</td>
      <td>26.705889</td>
      <td>-29.093593</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 52 columns</p>
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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5856</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5856</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2016-11-19 01:59:59</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Columns dengan nilai yang hilang:
    Column primary_cleaner.input.sulfate dengan 5.1571% persentasi nilai yang hilang , dan 302 nilai yang hilang
    Column primary_cleaner.input.depressant dengan 4.8497% persentasi nilai yang hilang , dan 284 nilai yang hilang
    Column primary_cleaner.input.xanthate dengan 2.8347% persentasi nilai yang hilang , dan 166 nilai yang hilang
    Column primary_cleaner.state.floatbank8_a_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_b_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_b_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_c_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_c_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_d_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column primary_cleaner.state.floatbank8_d_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.input.feed_ag dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.input.feed_pb dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.input.feed_rate dengan 0.6831% persentasi nilai yang hilang , dan 40 nilai yang hilang
    Column rougher.input.feed_size dengan 0.3757% persentasi nilai yang hilang , dan 22 nilai yang hilang
    Column rougher.input.feed_sol dengan 1.1441% persentasi nilai yang hilang , dan 67 nilai yang hilang
    Column rougher.input.feed_au dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.input.floatbank10_sulfate dengan 4.3887% persentasi nilai yang hilang , dan 257 nilai yang hilang
    Column rougher.input.floatbank10_xanthate dengan 2.1004% persentasi nilai yang hilang , dan 123 nilai yang hilang
    Column rougher.input.floatbank11_sulfate dengan 0.9392% persentasi nilai yang hilang , dan 55 nilai yang hilang
    Column rougher.input.floatbank11_xanthate dengan 6.0280% persentasi nilai yang hilang , dan 353 nilai yang hilang
    Column rougher.state.floatbank10_a_air dengan 0.2903% persentasi nilai yang hilang , dan 17 nilai yang hilang
    Column rougher.state.floatbank10_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.state.floatbank10_b_air dengan 0.2903% persentasi nilai yang hilang , dan 17 nilai yang hilang
    Column rougher.state.floatbank10_b_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.state.floatbank10_c_air dengan 0.2903% persentasi nilai yang hilang , dan 17 nilai yang hilang
    Column rougher.state.floatbank10_c_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.state.floatbank10_d_air dengan 0.2903% persentasi nilai yang hilang , dan 17 nilai yang hilang
    Column rougher.state.floatbank10_d_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.state.floatbank10_e_air dengan 0.2903% persentasi nilai yang hilang , dan 17 nilai yang hilang
    Column rougher.state.floatbank10_e_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column rougher.state.floatbank10_f_air dengan 0.2903% persentasi nilai yang hilang , dan 17 nilai yang hilang
    Column rougher.state.floatbank10_f_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_a_air dengan 0.3415% persentasi nilai yang hilang , dan 20 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_b_air dengan 0.3928% persentasi nilai yang hilang , dan 23 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_b_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_a_air dengan 0.5806% persentasi nilai yang hilang , dan 34 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_b_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_b_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_a_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_b_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_b_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_a_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_b_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_b_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank6_a_air dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    Column secondary_cleaner.state.floatbank6_a_level dengan 0.2732% persentasi nilai yang hilang , dan 16 nilai yang hilang
    [1mTerdapat 51 columns dengan nilai NA.[0m



    None


    ----------------------------------------------------------------------------------------------------
    Shape:
    (5856, 53)
    ----------------------------------------------------------------------------------------------------
    Duplicated:
    [1mKita mempunyai 0 baris yang terduplikasi.
    [0m
    



```python
print('Informasi Umum pada Dataset gold_recovery_full')
get_info(gold_recovery_full)
```

    Informasi Umum pada Dataset gold_recovery_full
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
      <th>date</th>
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-15 00:00:00</td>
      <td>6.055403</td>
      <td>9.889648</td>
      <td>5.507324</td>
      <td>42.192020</td>
      <td>70.541216</td>
      <td>10.411962</td>
      <td>0.895447</td>
      <td>16.904297</td>
      <td>2.143149</td>
      <td>...</td>
      <td>14.016835</td>
      <td>-502.488007</td>
      <td>12.099931</td>
      <td>-504.715942</td>
      <td>9.925633</td>
      <td>-498.310211</td>
      <td>8.079666</td>
      <td>-500.470978</td>
      <td>14.151341</td>
      <td>-605.841980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-15 01:00:00</td>
      <td>6.029369</td>
      <td>9.968944</td>
      <td>5.257781</td>
      <td>42.701629</td>
      <td>69.266198</td>
      <td>10.462676</td>
      <td>0.927452</td>
      <td>16.634514</td>
      <td>2.224930</td>
      <td>...</td>
      <td>13.992281</td>
      <td>-505.503262</td>
      <td>11.950531</td>
      <td>-501.331529</td>
      <td>10.039245</td>
      <td>-500.169983</td>
      <td>7.984757</td>
      <td>-500.582168</td>
      <td>13.998353</td>
      <td>-599.787184</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-15 02:00:00</td>
      <td>6.055926</td>
      <td>10.213995</td>
      <td>5.383759</td>
      <td>42.657501</td>
      <td>68.116445</td>
      <td>10.507046</td>
      <td>0.953716</td>
      <td>16.208849</td>
      <td>2.257889</td>
      <td>...</td>
      <td>14.015015</td>
      <td>-502.520901</td>
      <td>11.912783</td>
      <td>-501.133383</td>
      <td>10.070913</td>
      <td>-500.129135</td>
      <td>8.013877</td>
      <td>-500.517572</td>
      <td>14.028663</td>
      <td>-601.427363</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-15 03:00:00</td>
      <td>6.047977</td>
      <td>9.977019</td>
      <td>4.858634</td>
      <td>42.689819</td>
      <td>68.347543</td>
      <td>10.422762</td>
      <td>0.883763</td>
      <td>16.532835</td>
      <td>2.146849</td>
      <td>...</td>
      <td>14.036510</td>
      <td>-500.857308</td>
      <td>11.999550</td>
      <td>-501.193686</td>
      <td>9.970366</td>
      <td>-499.201640</td>
      <td>7.977324</td>
      <td>-500.255908</td>
      <td>14.005551</td>
      <td>-599.996129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-15 04:00:00</td>
      <td>6.148599</td>
      <td>10.142511</td>
      <td>4.939416</td>
      <td>42.774141</td>
      <td>66.927016</td>
      <td>10.360302</td>
      <td>0.792826</td>
      <td>16.525686</td>
      <td>2.055292</td>
      <td>...</td>
      <td>14.027298</td>
      <td>-499.838632</td>
      <td>11.953070</td>
      <td>-501.053894</td>
      <td>9.925709</td>
      <td>-501.686727</td>
      <td>7.894242</td>
      <td>-500.356035</td>
      <td>13.996647</td>
      <td>-601.496691</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 87 columns</p>
</div>


    ----------------------------------------------------------------------------------------------------
    Info:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22716 entries, 0 to 22715
    Data columns (total 87 columns):
     #   Column                                              Non-Null Count  Dtype  
    ---  ------                                              --------------  -----  
     0   date                                                22716 non-null  object 
     1   final.output.concentrate_ag                         22627 non-null  float64
     2   final.output.concentrate_pb                         22629 non-null  float64
     3   final.output.concentrate_sol                        22331 non-null  float64
     4   final.output.concentrate_au                         22630 non-null  float64
     5   final.output.recovery                               20753 non-null  float64
     6   final.output.tail_ag                                22633 non-null  float64
     7   final.output.tail_pb                                22516 non-null  float64
     8   final.output.tail_sol                               22445 non-null  float64
     9   final.output.tail_au                                22635 non-null  float64
     10  primary_cleaner.input.sulfate                       21107 non-null  float64
     11  primary_cleaner.input.depressant                    21170 non-null  float64
     12  primary_cleaner.input.feed_size                     22716 non-null  float64
     13  primary_cleaner.input.xanthate                      21565 non-null  float64
     14  primary_cleaner.output.concentrate_ag               22618 non-null  float64
     15  primary_cleaner.output.concentrate_pb               22268 non-null  float64
     16  primary_cleaner.output.concentrate_sol              21918 non-null  float64
     17  primary_cleaner.output.concentrate_au               22618 non-null  float64
     18  primary_cleaner.output.tail_ag                      22614 non-null  float64
     19  primary_cleaner.output.tail_pb                      22594 non-null  float64
     20  primary_cleaner.output.tail_sol                     22365 non-null  float64
     21  primary_cleaner.output.tail_au                      22617 non-null  float64
     22  primary_cleaner.state.floatbank8_a_air              22660 non-null  float64
     23  primary_cleaner.state.floatbank8_a_level            22667 non-null  float64
     24  primary_cleaner.state.floatbank8_b_air              22660 non-null  float64
     25  primary_cleaner.state.floatbank8_b_level            22673 non-null  float64
     26  primary_cleaner.state.floatbank8_c_air              22662 non-null  float64
     27  primary_cleaner.state.floatbank8_c_level            22673 non-null  float64
     28  primary_cleaner.state.floatbank8_d_air              22661 non-null  float64
     29  primary_cleaner.state.floatbank8_d_level            22673 non-null  float64
     30  rougher.calculation.sulfate_to_au_concentrate       22672 non-null  float64
     31  rougher.calculation.floatbank10_sulfate_to_au_feed  22672 non-null  float64
     32  rougher.calculation.floatbank11_sulfate_to_au_feed  22672 non-null  float64
     33  rougher.calculation.au_pb_ratio                     21089 non-null  float64
     34  rougher.input.feed_ag                               22618 non-null  float64
     35  rougher.input.feed_pb                               22472 non-null  float64
     36  rougher.input.feed_rate                             22163 non-null  float64
     37  rougher.input.feed_size                             22277 non-null  float64
     38  rougher.input.feed_sol                              22357 non-null  float64
     39  rougher.input.feed_au                               22617 non-null  float64
     40  rougher.input.floatbank10_sulfate                   21415 non-null  float64
     41  rougher.input.floatbank10_xanthate                  22247 non-null  float64
     42  rougher.input.floatbank11_sulfate                   22038 non-null  float64
     43  rougher.input.floatbank11_xanthate                  20459 non-null  float64
     44  rougher.output.concentrate_ag                       22618 non-null  float64
     45  rougher.output.concentrate_pb                       22618 non-null  float64
     46  rougher.output.concentrate_sol                      22526 non-null  float64
     47  rougher.output.concentrate_au                       22618 non-null  float64
     48  rougher.output.recovery                             19597 non-null  float64
     49  rougher.output.tail_ag                              19979 non-null  float64
     50  rougher.output.tail_pb                              22618 non-null  float64
     51  rougher.output.tail_sol                             19980 non-null  float64
     52  rougher.output.tail_au                              19980 non-null  float64
     53  rougher.state.floatbank10_a_air                     22646 non-null  float64
     54  rougher.state.floatbank10_a_level                   22647 non-null  float64
     55  rougher.state.floatbank10_b_air                     22646 non-null  float64
     56  rougher.state.floatbank10_b_level                   22647 non-null  float64
     57  rougher.state.floatbank10_c_air                     22646 non-null  float64
     58  rougher.state.floatbank10_c_level                   22654 non-null  float64
     59  rougher.state.floatbank10_d_air                     22641 non-null  float64
     60  rougher.state.floatbank10_d_level                   22649 non-null  float64
     61  rougher.state.floatbank10_e_air                     22096 non-null  float64
     62  rougher.state.floatbank10_e_level                   22649 non-null  float64
     63  rougher.state.floatbank10_f_air                     22641 non-null  float64
     64  rougher.state.floatbank10_f_level                   22642 non-null  float64
     65  secondary_cleaner.output.tail_ag                    22616 non-null  float64
     66  secondary_cleaner.output.tail_pb                    22600 non-null  float64
     67  secondary_cleaner.output.tail_sol                   20501 non-null  float64
     68  secondary_cleaner.output.tail_au                    22618 non-null  float64
     69  secondary_cleaner.state.floatbank2_a_air            22333 non-null  float64
     70  secondary_cleaner.state.floatbank2_a_level          22591 non-null  float64
     71  secondary_cleaner.state.floatbank2_b_air            22538 non-null  float64
     72  secondary_cleaner.state.floatbank2_b_level          22588 non-null  float64
     73  secondary_cleaner.state.floatbank3_a_air            22585 non-null  float64
     74  secondary_cleaner.state.floatbank3_a_level          22587 non-null  float64
     75  secondary_cleaner.state.floatbank3_b_air            22592 non-null  float64
     76  secondary_cleaner.state.floatbank3_b_level          22590 non-null  float64
     77  secondary_cleaner.state.floatbank4_a_air            22571 non-null  float64
     78  secondary_cleaner.state.floatbank4_a_level          22587 non-null  float64
     79  secondary_cleaner.state.floatbank4_b_air            22608 non-null  float64
     80  secondary_cleaner.state.floatbank4_b_level          22607 non-null  float64
     81  secondary_cleaner.state.floatbank5_a_air            22615 non-null  float64
     82  secondary_cleaner.state.floatbank5_a_level          22615 non-null  float64
     83  secondary_cleaner.state.floatbank5_b_air            22615 non-null  float64
     84  secondary_cleaner.state.floatbank5_b_level          22616 non-null  float64
     85  secondary_cleaner.state.floatbank6_a_air            22597 non-null  float64
     86  secondary_cleaner.state.floatbank6_a_level          22615 non-null  float64
    dtypes: float64(86), object(1)
    memory usage: 15.1+ MB



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
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>primary_cleaner.input.sulfate</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22627.000000</td>
      <td>22629.000000</td>
      <td>22331.000000</td>
      <td>22630.000000</td>
      <td>20753.000000</td>
      <td>22633.000000</td>
      <td>22516.000000</td>
      <td>22445.000000</td>
      <td>22635.000000</td>
      <td>21107.000000</td>
      <td>...</td>
      <td>22571.000000</td>
      <td>22587.000000</td>
      <td>22608.000000</td>
      <td>22607.000000</td>
      <td>22615.000000</td>
      <td>22615.000000</td>
      <td>22615.000000</td>
      <td>22616.000000</td>
      <td>22597.000000</td>
      <td>22615.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.781559</td>
      <td>9.095308</td>
      <td>8.640317</td>
      <td>40.001172</td>
      <td>67.447488</td>
      <td>8.923690</td>
      <td>2.488252</td>
      <td>9.523632</td>
      <td>2.827459</td>
      <td>140.277672</td>
      <td>...</td>
      <td>18.205125</td>
      <td>-499.878977</td>
      <td>14.356474</td>
      <td>-476.532613</td>
      <td>14.883276</td>
      <td>-503.323288</td>
      <td>11.626743</td>
      <td>-500.521502</td>
      <td>17.976810</td>
      <td>-519.361465</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.030128</td>
      <td>3.230797</td>
      <td>3.785035</td>
      <td>13.398062</td>
      <td>11.616034</td>
      <td>3.517917</td>
      <td>1.189407</td>
      <td>4.079739</td>
      <td>1.262834</td>
      <td>49.919004</td>
      <td>...</td>
      <td>6.560700</td>
      <td>80.273964</td>
      <td>5.655791</td>
      <td>93.822791</td>
      <td>6.372811</td>
      <td>72.925589</td>
      <td>5.757449</td>
      <td>78.956292</td>
      <td>6.636203</td>
      <td>75.477151</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000003</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-799.920713</td>
      <td>0.000000</td>
      <td>-800.836914</td>
      <td>-0.423260</td>
      <td>-799.741097</td>
      <td>0.427084</td>
      <td>-800.258209</td>
      <td>-0.079426</td>
      <td>-810.473526</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.018525</td>
      <td>8.750171</td>
      <td>7.116799</td>
      <td>42.383721</td>
      <td>63.282393</td>
      <td>7.684016</td>
      <td>1.805376</td>
      <td>8.143576</td>
      <td>2.303108</td>
      <td>110.177081</td>
      <td>...</td>
      <td>14.095940</td>
      <td>-500.896232</td>
      <td>10.882675</td>
      <td>-500.309169</td>
      <td>10.941299</td>
      <td>-500.628697</td>
      <td>8.037533</td>
      <td>-500.167897</td>
      <td>13.968418</td>
      <td>-500.981671</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.953729</td>
      <td>9.914519</td>
      <td>8.908792</td>
      <td>44.653436</td>
      <td>68.322258</td>
      <td>9.484369</td>
      <td>2.653001</td>
      <td>10.212998</td>
      <td>2.913794</td>
      <td>141.330501</td>
      <td>...</td>
      <td>18.007326</td>
      <td>-499.917108</td>
      <td>14.947646</td>
      <td>-499.612292</td>
      <td>14.859117</td>
      <td>-499.865158</td>
      <td>10.989756</td>
      <td>-499.951980</td>
      <td>18.004215</td>
      <td>-500.095463</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.862593</td>
      <td>10.929839</td>
      <td>10.705824</td>
      <td>46.111999</td>
      <td>72.950836</td>
      <td>11.084557</td>
      <td>3.287790</td>
      <td>11.860824</td>
      <td>3.555077</td>
      <td>174.049914</td>
      <td>...</td>
      <td>22.998194</td>
      <td>-498.361545</td>
      <td>17.977502</td>
      <td>-400.224147</td>
      <td>18.014914</td>
      <td>-498.489381</td>
      <td>14.001193</td>
      <td>-499.492354</td>
      <td>23.009704</td>
      <td>-499.526388</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.001945</td>
      <td>17.031899</td>
      <td>19.615720</td>
      <td>53.611374</td>
      <td>100.000000</td>
      <td>19.552149</td>
      <td>6.086532</td>
      <td>22.861749</td>
      <td>9.789625</td>
      <td>274.409626</td>
      <td>...</td>
      <td>60.000000</td>
      <td>-127.692333</td>
      <td>31.269706</td>
      <td>-6.506986</td>
      <td>63.116298</td>
      <td>-244.483566</td>
      <td>39.846228</td>
      <td>-120.190931</td>
      <td>54.876806</td>
      <td>-29.093593</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 86 columns</p>
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
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22716</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>22716</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2018-05-19 07:59:59</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Columns dengan nilai yang hilang:
    Column final.output.concentrate_ag dengan 0.3918% persentasi nilai yang hilang , dan 89 nilai yang hilang
    Column final.output.concentrate_pb dengan 0.3830% persentasi nilai yang hilang , dan 87 nilai yang hilang
    Column final.output.concentrate_sol dengan 1.6948% persentasi nilai yang hilang , dan 385 nilai yang hilang
    Column final.output.concentrate_au dengan 0.3786% persentasi nilai yang hilang , dan 86 nilai yang hilang
    Column final.output.recovery dengan 8.6415% persentasi nilai yang hilang , dan 1963 nilai yang hilang
    Column final.output.tail_ag dengan 0.3654% persentasi nilai yang hilang , dan 83 nilai yang hilang
    Column final.output.tail_pb dengan 0.8804% persentasi nilai yang hilang , dan 200 nilai yang hilang
    Column final.output.tail_sol dengan 1.1930% persentasi nilai yang hilang , dan 271 nilai yang hilang
    Column final.output.tail_au dengan 0.3566% persentasi nilai yang hilang , dan 81 nilai yang hilang
    Column primary_cleaner.input.sulfate dengan 7.0831% persentasi nilai yang hilang , dan 1609 nilai yang hilang
    Column primary_cleaner.input.depressant dengan 6.8058% persentasi nilai yang hilang , dan 1546 nilai yang hilang
    Column primary_cleaner.input.xanthate dengan 5.0669% persentasi nilai yang hilang , dan 1151 nilai yang hilang
    Column primary_cleaner.output.concentrate_ag dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column primary_cleaner.output.concentrate_pb dengan 1.9722% persentasi nilai yang hilang , dan 448 nilai yang hilang
    Column primary_cleaner.output.concentrate_sol dengan 3.5129% persentasi nilai yang hilang , dan 798 nilai yang hilang
    Column primary_cleaner.output.concentrate_au dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column primary_cleaner.output.tail_ag dengan 0.4490% persentasi nilai yang hilang , dan 102 nilai yang hilang
    Column primary_cleaner.output.tail_pb dengan 0.5371% persentasi nilai yang hilang , dan 122 nilai yang hilang
    Column primary_cleaner.output.tail_sol dengan 1.5452% persentasi nilai yang hilang , dan 351 nilai yang hilang
    Column primary_cleaner.output.tail_au dengan 0.4358% persentasi nilai yang hilang , dan 99 nilai yang hilang
    Column primary_cleaner.state.floatbank8_a_air dengan 0.2465% persentasi nilai yang hilang , dan 56 nilai yang hilang
    Column primary_cleaner.state.floatbank8_a_level dengan 0.2157% persentasi nilai yang hilang , dan 49 nilai yang hilang
    Column primary_cleaner.state.floatbank8_b_air dengan 0.2465% persentasi nilai yang hilang , dan 56 nilai yang hilang
    Column primary_cleaner.state.floatbank8_b_level dengan 0.1893% persentasi nilai yang hilang , dan 43 nilai yang hilang
    Column primary_cleaner.state.floatbank8_c_air dengan 0.2377% persentasi nilai yang hilang , dan 54 nilai yang hilang
    Column primary_cleaner.state.floatbank8_c_level dengan 0.1893% persentasi nilai yang hilang , dan 43 nilai yang hilang
    Column primary_cleaner.state.floatbank8_d_air dengan 0.2421% persentasi nilai yang hilang , dan 55 nilai yang hilang
    Column primary_cleaner.state.floatbank8_d_level dengan 0.1893% persentasi nilai yang hilang , dan 43 nilai yang hilang
    Column rougher.calculation.sulfate_to_au_concentrate dengan 0.1937% persentasi nilai yang hilang , dan 44 nilai yang hilang
    Column rougher.calculation.floatbank10_sulfate_to_au_feed dengan 0.1937% persentasi nilai yang hilang , dan 44 nilai yang hilang
    Column rougher.calculation.floatbank11_sulfate_to_au_feed dengan 0.1937% persentasi nilai yang hilang , dan 44 nilai yang hilang
    Column rougher.calculation.au_pb_ratio dengan 7.1624% persentasi nilai yang hilang , dan 1627 nilai yang hilang
    Column rougher.input.feed_ag dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column rougher.input.feed_pb dengan 1.0741% persentasi nilai yang hilang , dan 244 nilai yang hilang
    Column rougher.input.feed_rate dengan 2.4344% persentasi nilai yang hilang , dan 553 nilai yang hilang
    Column rougher.input.feed_size dengan 1.9326% persentasi nilai yang hilang , dan 439 nilai yang hilang
    Column rougher.input.feed_sol dengan 1.5804% persentasi nilai yang hilang , dan 359 nilai yang hilang
    Column rougher.input.feed_au dengan 0.4358% persentasi nilai yang hilang , dan 99 nilai yang hilang
    Column rougher.input.floatbank10_sulfate dengan 5.7272% persentasi nilai yang hilang , dan 1301 nilai yang hilang
    Column rougher.input.floatbank10_xanthate dengan 2.0646% persentasi nilai yang hilang , dan 469 nilai yang hilang
    Column rougher.input.floatbank11_sulfate dengan 2.9847% persentasi nilai yang hilang , dan 678 nilai yang hilang
    Column rougher.input.floatbank11_xanthate dengan 9.9357% persentasi nilai yang hilang , dan 2257 nilai yang hilang
    Column rougher.output.concentrate_ag dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column rougher.output.concentrate_pb dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column rougher.output.concentrate_sol dengan 0.8364% persentasi nilai yang hilang , dan 190 nilai yang hilang
    Column rougher.output.concentrate_au dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column rougher.output.recovery dengan 13.7304% persentasi nilai yang hilang , dan 3119 nilai yang hilang
    Column rougher.output.tail_ag dengan 12.0488% persentasi nilai yang hilang , dan 2737 nilai yang hilang
    Column rougher.output.tail_pb dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column rougher.output.tail_sol dengan 12.0444% persentasi nilai yang hilang , dan 2736 nilai yang hilang
    Column rougher.output.tail_au dengan 12.0444% persentasi nilai yang hilang , dan 2736 nilai yang hilang
    Column rougher.state.floatbank10_a_air dengan 0.3082% persentasi nilai yang hilang , dan 70 nilai yang hilang
    Column rougher.state.floatbank10_a_level dengan 0.3038% persentasi nilai yang hilang , dan 69 nilai yang hilang
    Column rougher.state.floatbank10_b_air dengan 0.3082% persentasi nilai yang hilang , dan 70 nilai yang hilang
    Column rougher.state.floatbank10_b_level dengan 0.3038% persentasi nilai yang hilang , dan 69 nilai yang hilang
    Column rougher.state.floatbank10_c_air dengan 0.3082% persentasi nilai yang hilang , dan 70 nilai yang hilang
    Column rougher.state.floatbank10_c_level dengan 0.2729% persentasi nilai yang hilang , dan 62 nilai yang hilang
    Column rougher.state.floatbank10_d_air dengan 0.3302% persentasi nilai yang hilang , dan 75 nilai yang hilang
    Column rougher.state.floatbank10_d_level dengan 0.2949% persentasi nilai yang hilang , dan 67 nilai yang hilang
    Column rougher.state.floatbank10_e_air dengan 2.7294% persentasi nilai yang hilang , dan 620 nilai yang hilang
    Column rougher.state.floatbank10_e_level dengan 0.2949% persentasi nilai yang hilang , dan 67 nilai yang hilang
    Column rougher.state.floatbank10_f_air dengan 0.3302% persentasi nilai yang hilang , dan 75 nilai yang hilang
    Column rougher.state.floatbank10_f_level dengan 0.3258% persentasi nilai yang hilang , dan 74 nilai yang hilang
    Column secondary_cleaner.output.tail_ag dengan 0.4402% persentasi nilai yang hilang , dan 100 nilai yang hilang
    Column secondary_cleaner.output.tail_pb dengan 0.5107% persentasi nilai yang hilang , dan 116 nilai yang hilang
    Column secondary_cleaner.output.tail_sol dengan 9.7508% persentasi nilai yang hilang , dan 2215 nilai yang hilang
    Column secondary_cleaner.output.tail_au dengan 0.4314% persentasi nilai yang hilang , dan 98 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_a_air dengan 1.6860% persentasi nilai yang hilang , dan 383 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_a_level dengan 0.5503% persentasi nilai yang hilang , dan 125 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_b_air dengan 0.7836% persentasi nilai yang hilang , dan 178 nilai yang hilang
    Column secondary_cleaner.state.floatbank2_b_level dengan 0.5635% persentasi nilai yang hilang , dan 128 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_a_air dengan 0.5767% persentasi nilai yang hilang , dan 131 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_a_level dengan 0.5679% persentasi nilai yang hilang , dan 129 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_b_air dengan 0.5459% persentasi nilai yang hilang , dan 124 nilai yang hilang
    Column secondary_cleaner.state.floatbank3_b_level dengan 0.5547% persentasi nilai yang hilang , dan 126 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_a_air dengan 0.6383% persentasi nilai yang hilang , dan 145 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_a_level dengan 0.5679% persentasi nilai yang hilang , dan 129 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_b_air dengan 0.4754% persentasi nilai yang hilang , dan 108 nilai yang hilang
    Column secondary_cleaner.state.floatbank4_b_level dengan 0.4798% persentasi nilai yang hilang , dan 109 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_a_air dengan 0.4446% persentasi nilai yang hilang , dan 101 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_a_level dengan 0.4446% persentasi nilai yang hilang , dan 101 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_b_air dengan 0.4446% persentasi nilai yang hilang , dan 101 nilai yang hilang
    Column secondary_cleaner.state.floatbank5_b_level dengan 0.4402% persentasi nilai yang hilang , dan 100 nilai yang hilang
    Column secondary_cleaner.state.floatbank6_a_air dengan 0.5239% persentasi nilai yang hilang , dan 119 nilai yang hilang
    Column secondary_cleaner.state.floatbank6_a_level dengan 0.4446% persentasi nilai yang hilang , dan 101 nilai yang hilang
    [1mTerdapat 85 columns dengan nilai NA.[0m



    None


    ----------------------------------------------------------------------------------------------------
    Shape:
    (22716, 87)
    ----------------------------------------------------------------------------------------------------
    Duplicated:
    [1mKita mempunyai 0 baris yang terduplikasi.
    [0m
    


Kesimpulan : 

Setelah Mempelajari Informasi Umum pada kesuluruhan data set, dapat disimpulkan :

1. pada dataset gold_recovery_train terdapat 16.860 rows, 86 columns dan 85 column dengan nilai yang hilan NA.
2. pada dataset gold_recovery_test terdapat 5.858 rows, 52 columns dan 51 column dengan nilai yang hilan NA.
3. pada dataset gold_recovery_full terdapat 22.716 rows, 86 columns dan 85 column dengan nilai yang hilan NA.

### Memriksa apakah pemulihan dihitung dengan benar. Dengan menggunakan dataset training, hitung pemulihan untuk fitur [rougher.output.recovery.] dan menemukan MAE (Mean Absolute Error) antara perhitungan dan nilai fitur.



```python
from IPython.display import Image
Image(url= 'https://pictures.s3.yandex.net/resources/Recovery_1576238822_1589899219.jpg',width = 500, height = 200 )
```




<img src="https://pictures.s3.yandex.net/resources/Recovery_1576238822_1589899219.jpg" width="500" height="200"/>




```python
# Data Ekstrasi Emas
C = gold_recovery_train['rougher.output.concentrate_au']
F = gold_recovery_train['rougher.input.feed_au']
T = gold_recovery_train['rougher.output.tail_au']

# Perhitungan Recovery
calculated_recovery = (C * (F - T)) / (F * (C - T)) * 100
print(calculated_recovery.sample(10))
```

    4855     60.396106
    8797           NaN
    4499     88.462037
    9931     76.655054
    7378     83.538549
    1763     77.391452
    12827    89.899717
    8332     80.156727
    16829    88.153491
    12751    93.072489
    dtype: float64



```python
# Perhitungn Mean Absolute Error 
mae_calc = pd.DataFrame({'calculated_recovery' : calculated_recovery, 
                         'output_recovery' : gold_recovery_train['rougher.output.recovery']}).dropna()

# Menampilkan perhitunga MAE
mae = mean_absolute_error(mae_calc.calculated_recovery, mae_calc.output_recovery)
print('Score Mean Absolute Error dari rougher.output.recovery : {:.2f}'.format(mae))
```

    Score Mean Absolute Error dari rougher.output.recovery : 0.00



```python
# Memeriksa sample dari perhitungan recovery dan output recovery 
mae_calc.sample(10)
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
      <th>calculated_recovery</th>
      <th>output_recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14867</th>
      <td>88.478982</td>
      <td>88.478982</td>
    </tr>
    <tr>
      <th>290</th>
      <td>78.741704</td>
      <td>78.741704</td>
    </tr>
    <tr>
      <th>9249</th>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>407</th>
      <td>82.024248</td>
      <td>82.024248</td>
    </tr>
    <tr>
      <th>11053</th>
      <td>91.366706</td>
      <td>91.366706</td>
    </tr>
    <tr>
      <th>1170</th>
      <td>42.413734</td>
      <td>42.413734</td>
    </tr>
    <tr>
      <th>6523</th>
      <td>99.254694</td>
      <td>99.254694</td>
    </tr>
    <tr>
      <th>15745</th>
      <td>95.639581</td>
      <td>95.639581</td>
    </tr>
    <tr>
      <th>7140</th>
      <td>81.572936</td>
      <td>81.572936</td>
    </tr>
    <tr>
      <th>4478</th>
      <td>90.167393</td>
      <td>90.167393</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan :

Dari perhitungan yang dilakukan, kita dapat melihat bahwa  calculated_recovery dan rougher.output.recovery memiliki nilai yang sama. dan score Mean Absolute Erronya adalah 0,0. Ini menunjukkan bahwa nilai yang dihitung dari proses pemulihan yang disimulasikan mirip dengan rougher.output.recovery.


### Memeriksa fitur yang tidak tersedia serta parameter dan tipe logam nya di set pengujian


```python
# Fitur yang tidak tersedia di dataset test 
features_diff = list(set(gold_recovery_train.columns).difference(gold_recovery_test.columns))
features_diff = pd.DataFrame(features_diff, columns= ['feature_diff_test'])
features_diff
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
      <th>feature_diff_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rougher.output.concentrate_sol</td>
    </tr>
    <tr>
      <th>1</th>
      <td>final.output.concentrate_pb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>final.output.tail_ag</td>
    </tr>
    <tr>
      <th>3</th>
      <td>primary_cleaner.output.tail_ag</td>
    </tr>
    <tr>
      <th>4</th>
      <td>final.output.concentrate_sol</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rougher.calculation.floatbank10_sulfate_to_au_...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rougher.output.tail_au</td>
    </tr>
    <tr>
      <th>7</th>
      <td>rougher.output.concentrate_ag</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rougher.output.tail_pb</td>
    </tr>
    <tr>
      <th>9</th>
      <td>secondary_cleaner.output.tail_sol</td>
    </tr>
    <tr>
      <th>10</th>
      <td>rougher.calculation.au_pb_ratio</td>
    </tr>
    <tr>
      <th>11</th>
      <td>primary_cleaner.output.concentrate_sol</td>
    </tr>
    <tr>
      <th>12</th>
      <td>primary_cleaner.output.tail_sol</td>
    </tr>
    <tr>
      <th>13</th>
      <td>secondary_cleaner.output.tail_pb</td>
    </tr>
    <tr>
      <th>14</th>
      <td>primary_cleaner.output.concentrate_ag</td>
    </tr>
    <tr>
      <th>15</th>
      <td>secondary_cleaner.output.tail_ag</td>
    </tr>
    <tr>
      <th>16</th>
      <td>rougher.output.recovery</td>
    </tr>
    <tr>
      <th>17</th>
      <td>primary_cleaner.output.concentrate_pb</td>
    </tr>
    <tr>
      <th>18</th>
      <td>final.output.tail_au</td>
    </tr>
    <tr>
      <th>19</th>
      <td>primary_cleaner.output.tail_au</td>
    </tr>
    <tr>
      <th>20</th>
      <td>rougher.output.tail_ag</td>
    </tr>
    <tr>
      <th>21</th>
      <td>rougher.output.concentrate_pb</td>
    </tr>
    <tr>
      <th>22</th>
      <td>rougher.calculation.floatbank11_sulfate_to_au_...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>rougher.output.tail_sol</td>
    </tr>
    <tr>
      <th>24</th>
      <td>final.output.tail_pb</td>
    </tr>
    <tr>
      <th>25</th>
      <td>final.output.recovery</td>
    </tr>
    <tr>
      <th>26</th>
      <td>rougher.output.concentrate_au</td>
    </tr>
    <tr>
      <th>27</th>
      <td>primary_cleaner.output.concentrate_au</td>
    </tr>
    <tr>
      <th>28</th>
      <td>final.output.tail_sol</td>
    </tr>
    <tr>
      <th>29</th>
      <td>rougher.calculation.sulfate_to_au_concentrate</td>
    </tr>
    <tr>
      <th>30</th>
      <td>primary_cleaner.output.tail_pb</td>
    </tr>
    <tr>
      <th>31</th>
      <td>final.output.concentrate_ag</td>
    </tr>
    <tr>
      <th>32</th>
      <td>secondary_cleaner.output.tail_au</td>
    </tr>
    <tr>
      <th>33</th>
      <td>final.output.concentrate_au</td>
    </tr>
  </tbody>
</table>
</div>




```python
from IPython.display import Image
Image(url= 'https://3.bp.blogspot.com/-WRvhf9RqPeY/W-esQxepAyI/AAAAAAAAJT0/rcr7BmoeJvUCdKdXwVFAgY6swwWFWA2wACLcBGAs/s1600/Unsur%2BPseudo%2Bgas%2BMulia.jpg',width = 500, height = 200 )
```




<img src="https://3.bp.blogspot.com/-WRvhf9RqPeY/W-esQxepAyI/AAAAAAAAJT0/rcr7BmoeJvUCdKdXwVFAgY6swwWFWA2wACLcBGAs/s1600/Unsur%2BPseudo%2Bgas%2BMulia.jpg" width="500" height="200"/>



Kesimpulan :

Setelah memeriksa fitur yang tidak tersedia di dataset test, dapat dilihat dataset train memiliki 34 fitur  yang tidak tersedia di dataset test. Fitur yang tidak terdapat pada dataset test antara lain fitur yang mengandung konsentrasi logam Au = emas, Ag = perak , Pb = timbal. dan jenis parameternya adalah output — product parameters, calculation — calculation characteristics


### Data Preprocessing.


```python
# Fungsi untuk menghitung persentase nilai yang hilang
def missing_values_table(df):

    # Total nilai yang hilang
    mis_val = df.isnull().sum()

    # Persentase Nilai yang hilang
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Membuat tabel dengan hasil
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Merubah nama colum
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'jumlah nilai yang hilang', 1 : 'persentase nilai yang hilang'})

    # Sort tabel berdasarkan persentase nilai yang hilang 
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    'persentase nilai yang hilang', ascending=False).round(1)

    # Menampilkan Informasi 
    print ("Dataset ini Memiliki " + str(df.shape[1]) + " columns.\n"      
        "Terdapat " + str(mis_val_table_ren_columns.shape[0]) +
            " columns yang memiliki nilai yang hilang." )
    print('-'*50)
    print(df.info())

    return mis_val_table_ren_columns
```


```python
missing_values_table(gold_recovery_train)
```

    Dataset ini Memiliki 87 columns.
    Terdapat 85 columns yang memiliki nilai yang hilang.
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16860 entries, 0 to 16859
    Data columns (total 87 columns):
     #   Column                                              Non-Null Count  Dtype  
    ---  ------                                              --------------  -----  
     0   date                                                16860 non-null  object 
     1   final.output.concentrate_ag                         16788 non-null  float64
     2   final.output.concentrate_pb                         16788 non-null  float64
     3   final.output.concentrate_sol                        16490 non-null  float64
     4   final.output.concentrate_au                         16789 non-null  float64
     5   final.output.recovery                               15339 non-null  float64
     6   final.output.tail_ag                                16794 non-null  float64
     7   final.output.tail_pb                                16677 non-null  float64
     8   final.output.tail_sol                               16715 non-null  float64
     9   final.output.tail_au                                16794 non-null  float64
     10  primary_cleaner.input.sulfate                       15553 non-null  float64
     11  primary_cleaner.input.depressant                    15598 non-null  float64
     12  primary_cleaner.input.feed_size                     16860 non-null  float64
     13  primary_cleaner.input.xanthate                      15875 non-null  float64
     14  primary_cleaner.output.concentrate_ag               16778 non-null  float64
     15  primary_cleaner.output.concentrate_pb               16502 non-null  float64
     16  primary_cleaner.output.concentrate_sol              16224 non-null  float64
     17  primary_cleaner.output.concentrate_au               16778 non-null  float64
     18  primary_cleaner.output.tail_ag                      16777 non-null  float64
     19  primary_cleaner.output.tail_pb                      16761 non-null  float64
     20  primary_cleaner.output.tail_sol                     16579 non-null  float64
     21  primary_cleaner.output.tail_au                      16777 non-null  float64
     22  primary_cleaner.state.floatbank8_a_air              16820 non-null  float64
     23  primary_cleaner.state.floatbank8_a_level            16827 non-null  float64
     24  primary_cleaner.state.floatbank8_b_air              16820 non-null  float64
     25  primary_cleaner.state.floatbank8_b_level            16833 non-null  float64
     26  primary_cleaner.state.floatbank8_c_air              16822 non-null  float64
     27  primary_cleaner.state.floatbank8_c_level            16833 non-null  float64
     28  primary_cleaner.state.floatbank8_d_air              16821 non-null  float64
     29  primary_cleaner.state.floatbank8_d_level            16833 non-null  float64
     30  rougher.calculation.sulfate_to_au_concentrate       16833 non-null  float64
     31  rougher.calculation.floatbank10_sulfate_to_au_feed  16833 non-null  float64
     32  rougher.calculation.floatbank11_sulfate_to_au_feed  16833 non-null  float64
     33  rougher.calculation.au_pb_ratio                     15618 non-null  float64
     34  rougher.input.feed_ag                               16778 non-null  float64
     35  rougher.input.feed_pb                               16632 non-null  float64
     36  rougher.input.feed_rate                             16347 non-null  float64
     37  rougher.input.feed_size                             16443 non-null  float64
     38  rougher.input.feed_sol                              16568 non-null  float64
     39  rougher.input.feed_au                               16777 non-null  float64
     40  rougher.input.floatbank10_sulfate                   15816 non-null  float64
     41  rougher.input.floatbank10_xanthate                  16514 non-null  float64
     42  rougher.input.floatbank11_sulfate                   16237 non-null  float64
     43  rougher.input.floatbank11_xanthate                  14956 non-null  float64
     44  rougher.output.concentrate_ag                       16778 non-null  float64
     45  rougher.output.concentrate_pb                       16778 non-null  float64
     46  rougher.output.concentrate_sol                      16698 non-null  float64
     47  rougher.output.concentrate_au                       16778 non-null  float64
     48  rougher.output.recovery                             14287 non-null  float64
     49  rougher.output.tail_ag                              14610 non-null  float64
     50  rougher.output.tail_pb                              16778 non-null  float64
     51  rougher.output.tail_sol                             14611 non-null  float64
     52  rougher.output.tail_au                              14611 non-null  float64
     53  rougher.state.floatbank10_a_air                     16807 non-null  float64
     54  rougher.state.floatbank10_a_level                   16807 non-null  float64
     55  rougher.state.floatbank10_b_air                     16807 non-null  float64
     56  rougher.state.floatbank10_b_level                   16807 non-null  float64
     57  rougher.state.floatbank10_c_air                     16807 non-null  float64
     58  rougher.state.floatbank10_c_level                   16814 non-null  float64
     59  rougher.state.floatbank10_d_air                     16802 non-null  float64
     60  rougher.state.floatbank10_d_level                   16809 non-null  float64
     61  rougher.state.floatbank10_e_air                     16257 non-null  float64
     62  rougher.state.floatbank10_e_level                   16809 non-null  float64
     63  rougher.state.floatbank10_f_air                     16802 non-null  float64
     64  rougher.state.floatbank10_f_level                   16802 non-null  float64
     65  secondary_cleaner.output.tail_ag                    16776 non-null  float64
     66  secondary_cleaner.output.tail_pb                    16764 non-null  float64
     67  secondary_cleaner.output.tail_sol                   14874 non-null  float64
     68  secondary_cleaner.output.tail_au                    16778 non-null  float64
     69  secondary_cleaner.state.floatbank2_a_air            16497 non-null  float64
     70  secondary_cleaner.state.floatbank2_a_level          16751 non-null  float64
     71  secondary_cleaner.state.floatbank2_b_air            16705 non-null  float64
     72  secondary_cleaner.state.floatbank2_b_level          16748 non-null  float64
     73  secondary_cleaner.state.floatbank3_a_air            16763 non-null  float64
     74  secondary_cleaner.state.floatbank3_a_level          16747 non-null  float64
     75  secondary_cleaner.state.floatbank3_b_air            16752 non-null  float64
     76  secondary_cleaner.state.floatbank3_b_level          16750 non-null  float64
     77  secondary_cleaner.state.floatbank4_a_air            16731 non-null  float64
     78  secondary_cleaner.state.floatbank4_a_level          16747 non-null  float64
     79  secondary_cleaner.state.floatbank4_b_air            16768 non-null  float64
     80  secondary_cleaner.state.floatbank4_b_level          16767 non-null  float64
     81  secondary_cleaner.state.floatbank5_a_air            16775 non-null  float64
     82  secondary_cleaner.state.floatbank5_a_level          16775 non-null  float64
     83  secondary_cleaner.state.floatbank5_b_air            16775 non-null  float64
     84  secondary_cleaner.state.floatbank5_b_level          16776 non-null  float64
     85  secondary_cleaner.state.floatbank6_a_air            16757 non-null  float64
     86  secondary_cleaner.state.floatbank6_a_level          16775 non-null  float64
    dtypes: float64(86), object(1)
    memory usage: 11.2+ MB
    None





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
      <th>jumlah nilai yang hilang</th>
      <th>persentase nilai yang hilang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rougher.output.recovery</th>
      <td>2573</td>
      <td>15.3</td>
    </tr>
    <tr>
      <th>rougher.output.tail_ag</th>
      <td>2250</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>rougher.output.tail_au</th>
      <td>2249</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>rougher.output.tail_sol</th>
      <td>2249</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.output.tail_sol</th>
      <td>1986</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_b_level</th>
      <td>27</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_c_level</th>
      <td>27</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_d_level</th>
      <td>27</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>rougher.calculation.sulfate_to_au_concentrate</th>
      <td>27</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>rougher.calculation.floatbank11_sulfate_to_au_feed</th>
      <td>27</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 2 columns</p>
</div>




```python
missing_values_table(gold_recovery_test)
```

    Dataset ini Memiliki 53 columns.
    Terdapat 51 columns yang memiliki nilai yang hilang.
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5856 entries, 0 to 5855
    Data columns (total 53 columns):
     #   Column                                      Non-Null Count  Dtype  
    ---  ------                                      --------------  -----  
     0   date                                        5856 non-null   object 
     1   primary_cleaner.input.sulfate               5554 non-null   float64
     2   primary_cleaner.input.depressant            5572 non-null   float64
     3   primary_cleaner.input.feed_size             5856 non-null   float64
     4   primary_cleaner.input.xanthate              5690 non-null   float64
     5   primary_cleaner.state.floatbank8_a_air      5840 non-null   float64
     6   primary_cleaner.state.floatbank8_a_level    5840 non-null   float64
     7   primary_cleaner.state.floatbank8_b_air      5840 non-null   float64
     8   primary_cleaner.state.floatbank8_b_level    5840 non-null   float64
     9   primary_cleaner.state.floatbank8_c_air      5840 non-null   float64
     10  primary_cleaner.state.floatbank8_c_level    5840 non-null   float64
     11  primary_cleaner.state.floatbank8_d_air      5840 non-null   float64
     12  primary_cleaner.state.floatbank8_d_level    5840 non-null   float64
     13  rougher.input.feed_ag                       5840 non-null   float64
     14  rougher.input.feed_pb                       5840 non-null   float64
     15  rougher.input.feed_rate                     5816 non-null   float64
     16  rougher.input.feed_size                     5834 non-null   float64
     17  rougher.input.feed_sol                      5789 non-null   float64
     18  rougher.input.feed_au                       5840 non-null   float64
     19  rougher.input.floatbank10_sulfate           5599 non-null   float64
     20  rougher.input.floatbank10_xanthate          5733 non-null   float64
     21  rougher.input.floatbank11_sulfate           5801 non-null   float64
     22  rougher.input.floatbank11_xanthate          5503 non-null   float64
     23  rougher.state.floatbank10_a_air             5839 non-null   float64
     24  rougher.state.floatbank10_a_level           5840 non-null   float64
     25  rougher.state.floatbank10_b_air             5839 non-null   float64
     26  rougher.state.floatbank10_b_level           5840 non-null   float64
     27  rougher.state.floatbank10_c_air             5839 non-null   float64
     28  rougher.state.floatbank10_c_level           5840 non-null   float64
     29  rougher.state.floatbank10_d_air             5839 non-null   float64
     30  rougher.state.floatbank10_d_level           5840 non-null   float64
     31  rougher.state.floatbank10_e_air             5839 non-null   float64
     32  rougher.state.floatbank10_e_level           5840 non-null   float64
     33  rougher.state.floatbank10_f_air             5839 non-null   float64
     34  rougher.state.floatbank10_f_level           5840 non-null   float64
     35  secondary_cleaner.state.floatbank2_a_air    5836 non-null   float64
     36  secondary_cleaner.state.floatbank2_a_level  5840 non-null   float64
     37  secondary_cleaner.state.floatbank2_b_air    5833 non-null   float64
     38  secondary_cleaner.state.floatbank2_b_level  5840 non-null   float64
     39  secondary_cleaner.state.floatbank3_a_air    5822 non-null   float64
     40  secondary_cleaner.state.floatbank3_a_level  5840 non-null   float64
     41  secondary_cleaner.state.floatbank3_b_air    5840 non-null   float64
     42  secondary_cleaner.state.floatbank3_b_level  5840 non-null   float64
     43  secondary_cleaner.state.floatbank4_a_air    5840 non-null   float64
     44  secondary_cleaner.state.floatbank4_a_level  5840 non-null   float64
     45  secondary_cleaner.state.floatbank4_b_air    5840 non-null   float64
     46  secondary_cleaner.state.floatbank4_b_level  5840 non-null   float64
     47  secondary_cleaner.state.floatbank5_a_air    5840 non-null   float64
     48  secondary_cleaner.state.floatbank5_a_level  5840 non-null   float64
     49  secondary_cleaner.state.floatbank5_b_air    5840 non-null   float64
     50  secondary_cleaner.state.floatbank5_b_level  5840 non-null   float64
     51  secondary_cleaner.state.floatbank6_a_air    5840 non-null   float64
     52  secondary_cleaner.state.floatbank6_a_level  5840 non-null   float64
    dtypes: float64(52), object(1)
    memory usage: 2.4+ MB
    None





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
      <th>jumlah nilai yang hilang</th>
      <th>persentase nilai yang hilang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rougher.input.floatbank11_xanthate</th>
      <td>353</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>primary_cleaner.input.sulfate</th>
      <td>302</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>primary_cleaner.input.depressant</th>
      <td>284</td>
      <td>4.8</td>
    </tr>
    <tr>
      <th>rougher.input.floatbank10_sulfate</th>
      <td>257</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>primary_cleaner.input.xanthate</th>
      <td>166</td>
      <td>2.8</td>
    </tr>
    <tr>
      <th>rougher.input.floatbank10_xanthate</th>
      <td>123</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>rougher.input.feed_sol</th>
      <td>67</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>rougher.input.floatbank11_sulfate</th>
      <td>55</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>rougher.input.feed_rate</th>
      <td>40</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank3_a_air</th>
      <td>34</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank2_b_air</th>
      <td>23</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>rougher.input.feed_size</th>
      <td>22</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank2_a_air</th>
      <td>20</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_f_air</th>
      <td>17</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_e_air</th>
      <td>17</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_d_air</th>
      <td>17</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_b_air</th>
      <td>17</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_a_air</th>
      <td>17</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_c_air</th>
      <td>17</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.input.feed_au</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.input.feed_pb</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank4_b_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank4_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank4_a_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank3_b_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank3_b_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank3_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_a_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank2_b_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank2_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_b_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_f_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_b_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_e_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_c_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_d_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_c_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_c_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_d_air</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_b_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_d_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.state.floatbank10_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>rougher.input.feed_ag</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
      <td>16</td>
      <td>0.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
missing_values_table(gold_recovery_full)
```

    Dataset ini Memiliki 87 columns.
    Terdapat 85 columns yang memiliki nilai yang hilang.
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22716 entries, 0 to 22715
    Data columns (total 87 columns):
     #   Column                                              Non-Null Count  Dtype  
    ---  ------                                              --------------  -----  
     0   date                                                22716 non-null  object 
     1   final.output.concentrate_ag                         22627 non-null  float64
     2   final.output.concentrate_pb                         22629 non-null  float64
     3   final.output.concentrate_sol                        22331 non-null  float64
     4   final.output.concentrate_au                         22630 non-null  float64
     5   final.output.recovery                               20753 non-null  float64
     6   final.output.tail_ag                                22633 non-null  float64
     7   final.output.tail_pb                                22516 non-null  float64
     8   final.output.tail_sol                               22445 non-null  float64
     9   final.output.tail_au                                22635 non-null  float64
     10  primary_cleaner.input.sulfate                       21107 non-null  float64
     11  primary_cleaner.input.depressant                    21170 non-null  float64
     12  primary_cleaner.input.feed_size                     22716 non-null  float64
     13  primary_cleaner.input.xanthate                      21565 non-null  float64
     14  primary_cleaner.output.concentrate_ag               22618 non-null  float64
     15  primary_cleaner.output.concentrate_pb               22268 non-null  float64
     16  primary_cleaner.output.concentrate_sol              21918 non-null  float64
     17  primary_cleaner.output.concentrate_au               22618 non-null  float64
     18  primary_cleaner.output.tail_ag                      22614 non-null  float64
     19  primary_cleaner.output.tail_pb                      22594 non-null  float64
     20  primary_cleaner.output.tail_sol                     22365 non-null  float64
     21  primary_cleaner.output.tail_au                      22617 non-null  float64
     22  primary_cleaner.state.floatbank8_a_air              22660 non-null  float64
     23  primary_cleaner.state.floatbank8_a_level            22667 non-null  float64
     24  primary_cleaner.state.floatbank8_b_air              22660 non-null  float64
     25  primary_cleaner.state.floatbank8_b_level            22673 non-null  float64
     26  primary_cleaner.state.floatbank8_c_air              22662 non-null  float64
     27  primary_cleaner.state.floatbank8_c_level            22673 non-null  float64
     28  primary_cleaner.state.floatbank8_d_air              22661 non-null  float64
     29  primary_cleaner.state.floatbank8_d_level            22673 non-null  float64
     30  rougher.calculation.sulfate_to_au_concentrate       22672 non-null  float64
     31  rougher.calculation.floatbank10_sulfate_to_au_feed  22672 non-null  float64
     32  rougher.calculation.floatbank11_sulfate_to_au_feed  22672 non-null  float64
     33  rougher.calculation.au_pb_ratio                     21089 non-null  float64
     34  rougher.input.feed_ag                               22618 non-null  float64
     35  rougher.input.feed_pb                               22472 non-null  float64
     36  rougher.input.feed_rate                             22163 non-null  float64
     37  rougher.input.feed_size                             22277 non-null  float64
     38  rougher.input.feed_sol                              22357 non-null  float64
     39  rougher.input.feed_au                               22617 non-null  float64
     40  rougher.input.floatbank10_sulfate                   21415 non-null  float64
     41  rougher.input.floatbank10_xanthate                  22247 non-null  float64
     42  rougher.input.floatbank11_sulfate                   22038 non-null  float64
     43  rougher.input.floatbank11_xanthate                  20459 non-null  float64
     44  rougher.output.concentrate_ag                       22618 non-null  float64
     45  rougher.output.concentrate_pb                       22618 non-null  float64
     46  rougher.output.concentrate_sol                      22526 non-null  float64
     47  rougher.output.concentrate_au                       22618 non-null  float64
     48  rougher.output.recovery                             19597 non-null  float64
     49  rougher.output.tail_ag                              19979 non-null  float64
     50  rougher.output.tail_pb                              22618 non-null  float64
     51  rougher.output.tail_sol                             19980 non-null  float64
     52  rougher.output.tail_au                              19980 non-null  float64
     53  rougher.state.floatbank10_a_air                     22646 non-null  float64
     54  rougher.state.floatbank10_a_level                   22647 non-null  float64
     55  rougher.state.floatbank10_b_air                     22646 non-null  float64
     56  rougher.state.floatbank10_b_level                   22647 non-null  float64
     57  rougher.state.floatbank10_c_air                     22646 non-null  float64
     58  rougher.state.floatbank10_c_level                   22654 non-null  float64
     59  rougher.state.floatbank10_d_air                     22641 non-null  float64
     60  rougher.state.floatbank10_d_level                   22649 non-null  float64
     61  rougher.state.floatbank10_e_air                     22096 non-null  float64
     62  rougher.state.floatbank10_e_level                   22649 non-null  float64
     63  rougher.state.floatbank10_f_air                     22641 non-null  float64
     64  rougher.state.floatbank10_f_level                   22642 non-null  float64
     65  secondary_cleaner.output.tail_ag                    22616 non-null  float64
     66  secondary_cleaner.output.tail_pb                    22600 non-null  float64
     67  secondary_cleaner.output.tail_sol                   20501 non-null  float64
     68  secondary_cleaner.output.tail_au                    22618 non-null  float64
     69  secondary_cleaner.state.floatbank2_a_air            22333 non-null  float64
     70  secondary_cleaner.state.floatbank2_a_level          22591 non-null  float64
     71  secondary_cleaner.state.floatbank2_b_air            22538 non-null  float64
     72  secondary_cleaner.state.floatbank2_b_level          22588 non-null  float64
     73  secondary_cleaner.state.floatbank3_a_air            22585 non-null  float64
     74  secondary_cleaner.state.floatbank3_a_level          22587 non-null  float64
     75  secondary_cleaner.state.floatbank3_b_air            22592 non-null  float64
     76  secondary_cleaner.state.floatbank3_b_level          22590 non-null  float64
     77  secondary_cleaner.state.floatbank4_a_air            22571 non-null  float64
     78  secondary_cleaner.state.floatbank4_a_level          22587 non-null  float64
     79  secondary_cleaner.state.floatbank4_b_air            22608 non-null  float64
     80  secondary_cleaner.state.floatbank4_b_level          22607 non-null  float64
     81  secondary_cleaner.state.floatbank5_a_air            22615 non-null  float64
     82  secondary_cleaner.state.floatbank5_a_level          22615 non-null  float64
     83  secondary_cleaner.state.floatbank5_b_air            22615 non-null  float64
     84  secondary_cleaner.state.floatbank5_b_level          22616 non-null  float64
     85  secondary_cleaner.state.floatbank6_a_air            22597 non-null  float64
     86  secondary_cleaner.state.floatbank6_a_level          22615 non-null  float64
    dtypes: float64(86), object(1)
    memory usage: 15.1+ MB
    None





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
      <th>jumlah nilai yang hilang</th>
      <th>persentase nilai yang hilang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rougher.output.recovery</th>
      <td>3119</td>
      <td>13.7</td>
    </tr>
    <tr>
      <th>rougher.output.tail_ag</th>
      <td>2737</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>rougher.output.tail_au</th>
      <td>2736</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>rougher.output.tail_sol</th>
      <td>2736</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>rougher.input.floatbank11_xanthate</th>
      <td>2257</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>rougher.calculation.floatbank10_sulfate_to_au_feed</th>
      <td>44</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>rougher.calculation.floatbank11_sulfate_to_au_feed</th>
      <td>44</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_d_level</th>
      <td>43</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_c_level</th>
      <td>43</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>primary_cleaner.state.floatbank8_b_level</th>
      <td>43</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 2 columns</p>
</div>




```python
# fungsi untuk mengubah column date menjadi tipe data datetime 
def convert_date_time(df, col):
    df[col] = pd.to_datetime(df[col])
    df.sort_values(col, inplace=True)
    
# fungsi untuk mengisi nilai yang hilang dengan metode 'ffill' 
def fill_missing_values(df):
    df = df.fillna(method='ffill', axis=0, inplace=True)
```


```python
# mengisi nilai yang hilang dengan metode 'ffill' 
fill_missing_values(gold_recovery_train)
fill_missing_values(gold_recovery_test)
fill_missing_values(gold_recovery_full)

# mengubah column date menjadi tipe data datetim
convert_date_time(gold_recovery_train, 'date')
convert_date_time(gold_recovery_test, 'date')
convert_date_time(gold_recovery_full, 'date')
```


```python
missing_values_table(gold_recovery_train)
```

    Dataset ini Memiliki 87 columns.
    Terdapat 0 columns yang memiliki nilai yang hilang.
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16860 entries, 0 to 16859
    Data columns (total 87 columns):
     #   Column                                              Non-Null Count  Dtype         
    ---  ------                                              --------------  -----         
     0   date                                                16860 non-null  datetime64[ns]
     1   final.output.concentrate_ag                         16860 non-null  float64       
     2   final.output.concentrate_pb                         16860 non-null  float64       
     3   final.output.concentrate_sol                        16860 non-null  float64       
     4   final.output.concentrate_au                         16860 non-null  float64       
     5   final.output.recovery                               16860 non-null  float64       
     6   final.output.tail_ag                                16860 non-null  float64       
     7   final.output.tail_pb                                16860 non-null  float64       
     8   final.output.tail_sol                               16860 non-null  float64       
     9   final.output.tail_au                                16860 non-null  float64       
     10  primary_cleaner.input.sulfate                       16860 non-null  float64       
     11  primary_cleaner.input.depressant                    16860 non-null  float64       
     12  primary_cleaner.input.feed_size                     16860 non-null  float64       
     13  primary_cleaner.input.xanthate                      16860 non-null  float64       
     14  primary_cleaner.output.concentrate_ag               16860 non-null  float64       
     15  primary_cleaner.output.concentrate_pb               16860 non-null  float64       
     16  primary_cleaner.output.concentrate_sol              16860 non-null  float64       
     17  primary_cleaner.output.concentrate_au               16860 non-null  float64       
     18  primary_cleaner.output.tail_ag                      16860 non-null  float64       
     19  primary_cleaner.output.tail_pb                      16860 non-null  float64       
     20  primary_cleaner.output.tail_sol                     16860 non-null  float64       
     21  primary_cleaner.output.tail_au                      16860 non-null  float64       
     22  primary_cleaner.state.floatbank8_a_air              16860 non-null  float64       
     23  primary_cleaner.state.floatbank8_a_level            16860 non-null  float64       
     24  primary_cleaner.state.floatbank8_b_air              16860 non-null  float64       
     25  primary_cleaner.state.floatbank8_b_level            16860 non-null  float64       
     26  primary_cleaner.state.floatbank8_c_air              16860 non-null  float64       
     27  primary_cleaner.state.floatbank8_c_level            16860 non-null  float64       
     28  primary_cleaner.state.floatbank8_d_air              16860 non-null  float64       
     29  primary_cleaner.state.floatbank8_d_level            16860 non-null  float64       
     30  rougher.calculation.sulfate_to_au_concentrate       16860 non-null  float64       
     31  rougher.calculation.floatbank10_sulfate_to_au_feed  16860 non-null  float64       
     32  rougher.calculation.floatbank11_sulfate_to_au_feed  16860 non-null  float64       
     33  rougher.calculation.au_pb_ratio                     16860 non-null  float64       
     34  rougher.input.feed_ag                               16860 non-null  float64       
     35  rougher.input.feed_pb                               16860 non-null  float64       
     36  rougher.input.feed_rate                             16860 non-null  float64       
     37  rougher.input.feed_size                             16860 non-null  float64       
     38  rougher.input.feed_sol                              16860 non-null  float64       
     39  rougher.input.feed_au                               16860 non-null  float64       
     40  rougher.input.floatbank10_sulfate                   16860 non-null  float64       
     41  rougher.input.floatbank10_xanthate                  16860 non-null  float64       
     42  rougher.input.floatbank11_sulfate                   16860 non-null  float64       
     43  rougher.input.floatbank11_xanthate                  16860 non-null  float64       
     44  rougher.output.concentrate_ag                       16860 non-null  float64       
     45  rougher.output.concentrate_pb                       16860 non-null  float64       
     46  rougher.output.concentrate_sol                      16860 non-null  float64       
     47  rougher.output.concentrate_au                       16860 non-null  float64       
     48  rougher.output.recovery                             16860 non-null  float64       
     49  rougher.output.tail_ag                              16860 non-null  float64       
     50  rougher.output.tail_pb                              16860 non-null  float64       
     51  rougher.output.tail_sol                             16860 non-null  float64       
     52  rougher.output.tail_au                              16860 non-null  float64       
     53  rougher.state.floatbank10_a_air                     16860 non-null  float64       
     54  rougher.state.floatbank10_a_level                   16860 non-null  float64       
     55  rougher.state.floatbank10_b_air                     16860 non-null  float64       
     56  rougher.state.floatbank10_b_level                   16860 non-null  float64       
     57  rougher.state.floatbank10_c_air                     16860 non-null  float64       
     58  rougher.state.floatbank10_c_level                   16860 non-null  float64       
     59  rougher.state.floatbank10_d_air                     16860 non-null  float64       
     60  rougher.state.floatbank10_d_level                   16860 non-null  float64       
     61  rougher.state.floatbank10_e_air                     16860 non-null  float64       
     62  rougher.state.floatbank10_e_level                   16860 non-null  float64       
     63  rougher.state.floatbank10_f_air                     16860 non-null  float64       
     64  rougher.state.floatbank10_f_level                   16860 non-null  float64       
     65  secondary_cleaner.output.tail_ag                    16860 non-null  float64       
     66  secondary_cleaner.output.tail_pb                    16860 non-null  float64       
     67  secondary_cleaner.output.tail_sol                   16860 non-null  float64       
     68  secondary_cleaner.output.tail_au                    16860 non-null  float64       
     69  secondary_cleaner.state.floatbank2_a_air            16860 non-null  float64       
     70  secondary_cleaner.state.floatbank2_a_level          16860 non-null  float64       
     71  secondary_cleaner.state.floatbank2_b_air            16860 non-null  float64       
     72  secondary_cleaner.state.floatbank2_b_level          16860 non-null  float64       
     73  secondary_cleaner.state.floatbank3_a_air            16860 non-null  float64       
     74  secondary_cleaner.state.floatbank3_a_level          16860 non-null  float64       
     75  secondary_cleaner.state.floatbank3_b_air            16860 non-null  float64       
     76  secondary_cleaner.state.floatbank3_b_level          16860 non-null  float64       
     77  secondary_cleaner.state.floatbank4_a_air            16860 non-null  float64       
     78  secondary_cleaner.state.floatbank4_a_level          16860 non-null  float64       
     79  secondary_cleaner.state.floatbank4_b_air            16860 non-null  float64       
     80  secondary_cleaner.state.floatbank4_b_level          16860 non-null  float64       
     81  secondary_cleaner.state.floatbank5_a_air            16860 non-null  float64       
     82  secondary_cleaner.state.floatbank5_a_level          16860 non-null  float64       
     83  secondary_cleaner.state.floatbank5_b_air            16860 non-null  float64       
     84  secondary_cleaner.state.floatbank5_b_level          16860 non-null  float64       
     85  secondary_cleaner.state.floatbank6_a_air            16860 non-null  float64       
     86  secondary_cleaner.state.floatbank6_a_level          16860 non-null  float64       
    dtypes: datetime64[ns](1), float64(86)
    memory usage: 11.3 MB
    None





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
      <th>jumlah nilai yang hilang</th>
      <th>persentase nilai yang hilang</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
missing_values_table(gold_recovery_test)
```

    Dataset ini Memiliki 53 columns.
    Terdapat 0 columns yang memiliki nilai yang hilang.
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5856 entries, 0 to 5855
    Data columns (total 53 columns):
     #   Column                                      Non-Null Count  Dtype         
    ---  ------                                      --------------  -----         
     0   date                                        5856 non-null   datetime64[ns]
     1   primary_cleaner.input.sulfate               5856 non-null   float64       
     2   primary_cleaner.input.depressant            5856 non-null   float64       
     3   primary_cleaner.input.feed_size             5856 non-null   float64       
     4   primary_cleaner.input.xanthate              5856 non-null   float64       
     5   primary_cleaner.state.floatbank8_a_air      5856 non-null   float64       
     6   primary_cleaner.state.floatbank8_a_level    5856 non-null   float64       
     7   primary_cleaner.state.floatbank8_b_air      5856 non-null   float64       
     8   primary_cleaner.state.floatbank8_b_level    5856 non-null   float64       
     9   primary_cleaner.state.floatbank8_c_air      5856 non-null   float64       
     10  primary_cleaner.state.floatbank8_c_level    5856 non-null   float64       
     11  primary_cleaner.state.floatbank8_d_air      5856 non-null   float64       
     12  primary_cleaner.state.floatbank8_d_level    5856 non-null   float64       
     13  rougher.input.feed_ag                       5856 non-null   float64       
     14  rougher.input.feed_pb                       5856 non-null   float64       
     15  rougher.input.feed_rate                     5856 non-null   float64       
     16  rougher.input.feed_size                     5856 non-null   float64       
     17  rougher.input.feed_sol                      5856 non-null   float64       
     18  rougher.input.feed_au                       5856 non-null   float64       
     19  rougher.input.floatbank10_sulfate           5856 non-null   float64       
     20  rougher.input.floatbank10_xanthate          5856 non-null   float64       
     21  rougher.input.floatbank11_sulfate           5856 non-null   float64       
     22  rougher.input.floatbank11_xanthate          5856 non-null   float64       
     23  rougher.state.floatbank10_a_air             5856 non-null   float64       
     24  rougher.state.floatbank10_a_level           5856 non-null   float64       
     25  rougher.state.floatbank10_b_air             5856 non-null   float64       
     26  rougher.state.floatbank10_b_level           5856 non-null   float64       
     27  rougher.state.floatbank10_c_air             5856 non-null   float64       
     28  rougher.state.floatbank10_c_level           5856 non-null   float64       
     29  rougher.state.floatbank10_d_air             5856 non-null   float64       
     30  rougher.state.floatbank10_d_level           5856 non-null   float64       
     31  rougher.state.floatbank10_e_air             5856 non-null   float64       
     32  rougher.state.floatbank10_e_level           5856 non-null   float64       
     33  rougher.state.floatbank10_f_air             5856 non-null   float64       
     34  rougher.state.floatbank10_f_level           5856 non-null   float64       
     35  secondary_cleaner.state.floatbank2_a_air    5856 non-null   float64       
     36  secondary_cleaner.state.floatbank2_a_level  5856 non-null   float64       
     37  secondary_cleaner.state.floatbank2_b_air    5856 non-null   float64       
     38  secondary_cleaner.state.floatbank2_b_level  5856 non-null   float64       
     39  secondary_cleaner.state.floatbank3_a_air    5856 non-null   float64       
     40  secondary_cleaner.state.floatbank3_a_level  5856 non-null   float64       
     41  secondary_cleaner.state.floatbank3_b_air    5856 non-null   float64       
     42  secondary_cleaner.state.floatbank3_b_level  5856 non-null   float64       
     43  secondary_cleaner.state.floatbank4_a_air    5856 non-null   float64       
     44  secondary_cleaner.state.floatbank4_a_level  5856 non-null   float64       
     45  secondary_cleaner.state.floatbank4_b_air    5856 non-null   float64       
     46  secondary_cleaner.state.floatbank4_b_level  5856 non-null   float64       
     47  secondary_cleaner.state.floatbank5_a_air    5856 non-null   float64       
     48  secondary_cleaner.state.floatbank5_a_level  5856 non-null   float64       
     49  secondary_cleaner.state.floatbank5_b_air    5856 non-null   float64       
     50  secondary_cleaner.state.floatbank5_b_level  5856 non-null   float64       
     51  secondary_cleaner.state.floatbank6_a_air    5856 non-null   float64       
     52  secondary_cleaner.state.floatbank6_a_level  5856 non-null   float64       
    dtypes: datetime64[ns](1), float64(52)
    memory usage: 2.4 MB
    None





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
      <th>jumlah nilai yang hilang</th>
      <th>persentase nilai yang hilang</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
missing_values_table(gold_recovery_full)
```

    Dataset ini Memiliki 87 columns.
    Terdapat 0 columns yang memiliki nilai yang hilang.
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 22716 entries, 0 to 22715
    Data columns (total 87 columns):
     #   Column                                              Non-Null Count  Dtype         
    ---  ------                                              --------------  -----         
     0   date                                                22716 non-null  datetime64[ns]
     1   final.output.concentrate_ag                         22716 non-null  float64       
     2   final.output.concentrate_pb                         22716 non-null  float64       
     3   final.output.concentrate_sol                        22716 non-null  float64       
     4   final.output.concentrate_au                         22716 non-null  float64       
     5   final.output.recovery                               22716 non-null  float64       
     6   final.output.tail_ag                                22716 non-null  float64       
     7   final.output.tail_pb                                22716 non-null  float64       
     8   final.output.tail_sol                               22716 non-null  float64       
     9   final.output.tail_au                                22716 non-null  float64       
     10  primary_cleaner.input.sulfate                       22716 non-null  float64       
     11  primary_cleaner.input.depressant                    22716 non-null  float64       
     12  primary_cleaner.input.feed_size                     22716 non-null  float64       
     13  primary_cleaner.input.xanthate                      22716 non-null  float64       
     14  primary_cleaner.output.concentrate_ag               22716 non-null  float64       
     15  primary_cleaner.output.concentrate_pb               22716 non-null  float64       
     16  primary_cleaner.output.concentrate_sol              22716 non-null  float64       
     17  primary_cleaner.output.concentrate_au               22716 non-null  float64       
     18  primary_cleaner.output.tail_ag                      22716 non-null  float64       
     19  primary_cleaner.output.tail_pb                      22716 non-null  float64       
     20  primary_cleaner.output.tail_sol                     22716 non-null  float64       
     21  primary_cleaner.output.tail_au                      22716 non-null  float64       
     22  primary_cleaner.state.floatbank8_a_air              22716 non-null  float64       
     23  primary_cleaner.state.floatbank8_a_level            22716 non-null  float64       
     24  primary_cleaner.state.floatbank8_b_air              22716 non-null  float64       
     25  primary_cleaner.state.floatbank8_b_level            22716 non-null  float64       
     26  primary_cleaner.state.floatbank8_c_air              22716 non-null  float64       
     27  primary_cleaner.state.floatbank8_c_level            22716 non-null  float64       
     28  primary_cleaner.state.floatbank8_d_air              22716 non-null  float64       
     29  primary_cleaner.state.floatbank8_d_level            22716 non-null  float64       
     30  rougher.calculation.sulfate_to_au_concentrate       22716 non-null  float64       
     31  rougher.calculation.floatbank10_sulfate_to_au_feed  22716 non-null  float64       
     32  rougher.calculation.floatbank11_sulfate_to_au_feed  22716 non-null  float64       
     33  rougher.calculation.au_pb_ratio                     22716 non-null  float64       
     34  rougher.input.feed_ag                               22716 non-null  float64       
     35  rougher.input.feed_pb                               22716 non-null  float64       
     36  rougher.input.feed_rate                             22716 non-null  float64       
     37  rougher.input.feed_size                             22716 non-null  float64       
     38  rougher.input.feed_sol                              22716 non-null  float64       
     39  rougher.input.feed_au                               22716 non-null  float64       
     40  rougher.input.floatbank10_sulfate                   22716 non-null  float64       
     41  rougher.input.floatbank10_xanthate                  22716 non-null  float64       
     42  rougher.input.floatbank11_sulfate                   22716 non-null  float64       
     43  rougher.input.floatbank11_xanthate                  22716 non-null  float64       
     44  rougher.output.concentrate_ag                       22716 non-null  float64       
     45  rougher.output.concentrate_pb                       22716 non-null  float64       
     46  rougher.output.concentrate_sol                      22716 non-null  float64       
     47  rougher.output.concentrate_au                       22716 non-null  float64       
     48  rougher.output.recovery                             22716 non-null  float64       
     49  rougher.output.tail_ag                              22716 non-null  float64       
     50  rougher.output.tail_pb                              22716 non-null  float64       
     51  rougher.output.tail_sol                             22716 non-null  float64       
     52  rougher.output.tail_au                              22716 non-null  float64       
     53  rougher.state.floatbank10_a_air                     22716 non-null  float64       
     54  rougher.state.floatbank10_a_level                   22716 non-null  float64       
     55  rougher.state.floatbank10_b_air                     22716 non-null  float64       
     56  rougher.state.floatbank10_b_level                   22716 non-null  float64       
     57  rougher.state.floatbank10_c_air                     22716 non-null  float64       
     58  rougher.state.floatbank10_c_level                   22716 non-null  float64       
     59  rougher.state.floatbank10_d_air                     22716 non-null  float64       
     60  rougher.state.floatbank10_d_level                   22716 non-null  float64       
     61  rougher.state.floatbank10_e_air                     22716 non-null  float64       
     62  rougher.state.floatbank10_e_level                   22716 non-null  float64       
     63  rougher.state.floatbank10_f_air                     22716 non-null  float64       
     64  rougher.state.floatbank10_f_level                   22716 non-null  float64       
     65  secondary_cleaner.output.tail_ag                    22716 non-null  float64       
     66  secondary_cleaner.output.tail_pb                    22716 non-null  float64       
     67  secondary_cleaner.output.tail_sol                   22716 non-null  float64       
     68  secondary_cleaner.output.tail_au                    22716 non-null  float64       
     69  secondary_cleaner.state.floatbank2_a_air            22716 non-null  float64       
     70  secondary_cleaner.state.floatbank2_a_level          22716 non-null  float64       
     71  secondary_cleaner.state.floatbank2_b_air            22716 non-null  float64       
     72  secondary_cleaner.state.floatbank2_b_level          22716 non-null  float64       
     73  secondary_cleaner.state.floatbank3_a_air            22716 non-null  float64       
     74  secondary_cleaner.state.floatbank3_a_level          22716 non-null  float64       
     75  secondary_cleaner.state.floatbank3_b_air            22716 non-null  float64       
     76  secondary_cleaner.state.floatbank3_b_level          22716 non-null  float64       
     77  secondary_cleaner.state.floatbank4_a_air            22716 non-null  float64       
     78  secondary_cleaner.state.floatbank4_a_level          22716 non-null  float64       
     79  secondary_cleaner.state.floatbank4_b_air            22716 non-null  float64       
     80  secondary_cleaner.state.floatbank4_b_level          22716 non-null  float64       
     81  secondary_cleaner.state.floatbank5_a_air            22716 non-null  float64       
     82  secondary_cleaner.state.floatbank5_a_level          22716 non-null  float64       
     83  secondary_cleaner.state.floatbank5_b_air            22716 non-null  float64       
     84  secondary_cleaner.state.floatbank5_b_level          22716 non-null  float64       
     85  secondary_cleaner.state.floatbank6_a_air            22716 non-null  float64       
     86  secondary_cleaner.state.floatbank6_a_level          22716 non-null  float64       
    dtypes: datetime64[ns](1), float64(86)
    memory usage: 15.3 MB
    None





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
      <th>jumlah nilai yang hilang</th>
      <th>persentase nilai yang hilang</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Kesimpulan : 

Pada tahap ini, kita memeriksa data dan mengamati bahwa : 
1. data gold_recovery_train, memiliki 16.860 rows, 86 features, dan 85 columns ngan NA. 
2. data gold_recovery_test , memiliki 5.856 rows, 52 features, dan 51 columns dengan NA. 
3. data gold_recovery_full , memiliki 22.716 baris, 86 features, dan 85 columns dengan NA. 

lalu setelah memeriksa dataset kita mendapati bahwa features 'date' memiliki tipe data yang tidak tepat. setelah merubah tipe data ke tipe data datetime kita mengisi nilai yang hilang dengan metode ffill metode ini mengganti nilai yang hilang dengan nilai dari baris sebelumnya (atau kolom sebelumnya, jika parameter sumbu diatur ke 'kolom').

## Menganalisa Data

### Analisis bagaimana konsentrasi logam (Au, Ag, Pb) berubah tergantung pada tahap pemurnian 

Untuk mengetahui bagaimana konsentrasi logam (Au, Ag, Pb) berubah tergantung pada tahap purification, kita memplot distribusi konsentrasi di setiap tahap.


```python
# Membuat variabel untuk perhitungan konsentrasi logam distribusi
metals = ['au', 'ag', 'pb']
stage_parameter = ['rougher.output.concentrate', 'primary_cleaner.output.concentrate', 'final.output.concentrate']
xcolors = ['green', 'orange', 'dodgerblue']

# Fungsi untuk membuat plot distribusi dari setiao konsentrasi logam
def plot_distribution(df):
    kwargs = dict(hist_kws={'alpha': 0.5}, kde_kws={'linewidth':2})
    for element in metals:
        plt.figure(figsize=(10,6), dpi=80)
        for features, colour in zip(stage_parameter, xcolors):
            sns.distplot(df[features+'_'+element], color=colour, label=features+'_'+element, **kwargs)
            plt.axvline(0, c="r")
        plt.title('Distribusi dari ' +element+ ' Konsentrasi di seluruh tahap purification')
        plt.xlabel('Konsentrasi dari '+element)
        plt.legend();
```


```python
# Menjalankan fungsi plot distribusi
plot_distribution(gold_recovery_train)
```


    
![png](output_39_0.png)
    



    
![png](output_39_1.png)
    



    
![png](output_39_2.png)
    


Kesimpulan

Plot di atas menunjukkan distribusi konsentrasi logam di seluruh tahap pemurnian. Melihat plot, kita mengamati bahwa konsentrasi emas (au) meningkat dari proses the rougher.output ke final.output. Hal ini menunjukkan bahwa proses berjalan sebagaimana mestinya. Konsentrasi logam lain menurun dalam kasus perak (ag) atau tetap hampir sama seperti timbal (pb).

### Membandingkan distribusi ukuran partikel feed di dataset train dan di dataset test. Jika distribusi bervariasi secara signifikan, evaluasi model akan salah.


```python
# fungsi untuk memplot distribusi dari partikel feed 
def plot_particle_size_distribution(df):
    kwargs = dict(hist_kws={'alpha': 0.5}, kde_kws={'linewidth':2})
    plt.figure(figsize=(10,6), dpi=80)
    for features, colour, labels in zip(input_feed, xcolors, xlabel_):
        sns.distplot(features, color=colour, label=labels, **kwargs)
        plt.axvline(0, c="r")
    plt.title('Distribusi dari partikel feed untuk '+ [x.split('__', 1)[1] for x in xlabel_][0])
    plt.xlabel('Distribusi Ukuran Partikeel Feed')
    plt.legend();
```


```python
# Rata-rata ukuran partikel untuk train dataset
rougher_input_train = gold_recovery_train['rougher.input.feed_size'].mean()
rougher_input_test = gold_recovery_test['rougher.input.feed_size'].mean()

# Rata-rata ukuran partikel untuk test dataset 
primary_cleaner_input_train, primary_cleaner_input_test = gold_recovery_train['primary_cleaner.input.feed_size'].mean(),gold_recovery_test['primary_cleaner.input.feed_size'].mean()

print('Rata-rata ukuran partikel untuk proses rougher pada train dataset {:.2f}'.format(rougher_input_train))
print('Rata-rata ukuran partikel untuk proses primary cleaner pada train dataset {:.2f}'.format(primary_cleaner_input_train))
print()
print('Rata-rata ukuran partikel untuk proses rougher pada dataset test  {:.2f}'.format(rougher_input_test))
print('Rata-rata ukuran partikel untuk proses primary cleaner pada dataset test {:.2f}'.format(primary_cleaner_input_test))
```

    Rata-rata ukuran partikel untuk proses rougher pada train dataset 60.19
    Rata-rata ukuran partikel untuk proses primary cleaner pada train dataset 7.30
    
    Rata-rata ukuran partikel untuk proses rougher pada dataset test  55.90
    Rata-rata ukuran partikel untuk proses primary cleaner pada dataset test 7.26


# Distribusi plot dari partikel feed dari proses 'rougher.input.feed_size'


```python
input_feed = [gold_recovery_train['rougher.input.feed_size'], gold_recovery_test['rougher.input.feed_size']]
xlabel_ = ['gold_recovery_train__rougher.input.feed_size', 'gold_recovery_test__rougher.input.feed_size']
xcolors = ['green', 'orange']

# distribusi dari ukuran partikel feed (rougher.input.feed_size)
plot_particle_size_distribution(gold_recovery_train)
```


    
![png](output_45_0.png)
    


# Distribusi plot dari partikel feed dari proses 'primary_cleaner.input.feed_size'


```python
input_feed = [gold_recovery_train['primary_cleaner.input.feed_size'], gold_recovery_test['primary_cleaner.input.feed_size']]
xlabel_ = ['gold_recovery_train__primary_cleaner.input.feed_size', 'gold_recovery_test__primary_cleaner.input.feed_size']
xcolors = ['dodgerblue', 'orange']

# distribusi dari ukuran partikel feed (primary_cleaner.input.feed_size)
plot_particle_size_distribution(gold_recovery_test)
```


    
![png](output_47_0.png)
    


Kesimpulan :

Pada Tahap ini kita membandingkan ukuran partikel rata-rata untuk dataset train dan test. disini kita mengamati bahwa ukuran rata-rata partikel sama. Contohnya, ukuran partikel feed rata-rata untuk dataset train dan test masing-masing yakni 60 dan 56 pada proses rougher.input.feed_size, dan rata-rata ukuran partikel feed untuk dataset train dan test yakni 7.30 dan 7.2 untuk proses primary_cleaner.input.feed_size. Dari hasil pengamatan distribusi partikel feed adalah sama, Artinya Distribusi dari dataset train dan test tidak bervariasi secara signifikan, pada tahap selanjutnya kita dapat melatih model di dataset train dan menguji model di dataset test. 


### Menganalisa nilai abnormal pada distribusi gold_recovery_full  distribusi dalam proses : raw feed, rougher concentrate, and final concentrate. apakah terdapat anomali / outlier pada semua distribusi? bagaimana langkah kita untuk mengeliminasinya jika terdapat anomali / outlier ada distribusi tersebut?



```python
# rougher.input.feed pada dataset full
gold_recovery_full['rougher.input.feed.total_concentration'] = gold_recovery_full[['rougher.input.feed_ag',
                                                                                   'rougher.input.feed_pb', 
                                                                                   'rougher.input.feed_sol', 
                                                                                   'rougher.input.feed_au']].sum(axis=1)

# memriksa nilai anomali
total_conc_stage_1 = gold_recovery_full['rougher.input.feed.total_concentration']
plt.figure(figsize=(10,6))
plt.hist(total_conc_stage_1, density=True, bins=100)

# menambahkan judul dan nama 
plt.xlabel('rougher.input.feed_total concentration')
plt.ylabel('frequency')
plt.title(" Total Distribusi dari proses konsentrasi pada raw feed");

```


    
![png](output_50_0.png)
    


Pada histogram di atas menggambarkan total distribusi konsentrasi proses raw feed. pada distribusi dari `[rougher.input.feed.total_concentration]` ini distribusi miring ke kiri dengan ekor panjang memanjang ke kiri dan sebagian besar nilai mengelompok di sebelah kanan. Pada tanda nol, kita melihat lonjakan besar. Ini merupakan anomali atau outlier dalam data.


```python
# rougher.output.concentrate pada full dataset
gold_recovery_full['rougher.output.concentrate.total_concentration'] = gold_recovery_full[['rougher.output.concentrate_ag',
                                                                                           'rougher.output.concentrate_pb',
                                                                                           'rougher.output.concentrate_sol',
                                                                                           'rougher.output.concentrate_au']].sum(axis=1)

# memriksa nilai anomali
total_conc_stage_2 = gold_recovery_full['rougher.output.concentrate.total_concentration']
plt.figure(figsize=(10,6))
plt.hist(total_conc_stage_2, density=True, bins=100)

# menambahkan judul dan nama 
plt.xlabel('rougher.output.concentrate_total concentration')
plt.ylabel('frequency')
plt.title("Total Distribusi dari proses konsentrasi pada rougher concentrate");
```


    
![png](output_52_0.png)
    


Pada histogram di atas menggambarkan total distribusi konsentrasi proses `rougher concentration`. pada distribusi dari `['rougher.output.concentrate.total_concentration']` ini distribusi miring ke kiri dengan ekor panjang memanjang ke kiri dan sebagian besar nilai mengelompok di sebelah kanan. Pada tanda nol, kita melihat lonjakan besar. Ini merupakan anomali atau outlier dalam data.


```python
# final.output.concentrate pada full dataset
gold_recovery_full['final.output.concentrate.total_concentration'] = gold_recovery_full[['rougher.input.feed_ag',
                                                                                   'rougher.input.feed_pb', 
                                                                                   'rougher.input.feed_sol', 
                                                                                   'rougher.input.feed_au']].sum(axis=1)

# memriksa nilai anomali
total_conc_stage_3 = gold_recovery_full['final.output.concentrate.total_concentration'] 
plt.figure(figsize=(10,6))
plt.hist(total_conc_stage_3, density=True, bins=100)

# menambahkan judul dan nama 
plt.xlabel('final.output.concentrate_total concentration')
plt.ylabel('frequency')
plt.title("Total Distribusi dari proses konsentrasi pada final concentrate");
```


    
![png](output_54_0.png)
    


Pada histogram di atas menggambarkan total distribusi konsentrasi proses `final concentrate`. pada distribusi dari `['final.output.concentrate.total_concentration']` ini distribusi miring ke kiri dengan ekor panjang memanjang ke kiri dan sebagian besar nilai mengelompok di sebelah kanan. Pada tanda nol, kita melihat lonjakan besar. Ini merupakan anomali atau outlier dalam data.

Kesimpulan :

Kami menjumlahkan konsentrasi total semua zat pada tahapan yang berbeda dalam proses yaitu. raw feed `[rougher.input.feed.total_concentration]`, rougher concetrate `[rougher.output.concentrate.total_concentration]`, dan final concentrate  `[final.output.concentrate.total_concentration]`. lalu dengan memplot histogram dan mengamati puncak di ketiga plot menunjukkan anomali dalam data. Sangat aneh jika memiliki konsentrasi total sekitar 0, sehingga puncaknya adalah nilai abnormal dan Salah satu alasan yang mungkin untuk anomali tersebut adalah kesalahan entri data atau mewakili pengamatan yang terjadi dalam kondisi yang tidak biasa. Atau, mungkin itu adalah oservasiyang mungkin secara akurat menggambarkan variabilitas di wilayah studi. kita tidak pernah tahu tetapi satu hal yang harus dilakukan adalah menghapus anomali ini agar tidak menimbulkan bias, dengan menghilangkannya dan menetapkan ambang batas 0,95. 

### Memfilter Full Dataset tanpa anomaly


```python
# memfilter full dataset tanpa anomaly 
gold_recovery_full_data =  gold_recovery_full[(gold_recovery_full['rougher.input.feed.total_concentration'] > 0.95) & 
                                        (gold_recovery_full['rougher.output.concentrate.total_concentration'] > 0.95) & 
                                        (gold_recovery_full['final.output.concentrate.total_concentration'] > 0.95)]

print('Panjang baris dari dataset full tanpa anomaly : ', format(gold_recovery_full_data.shape))

```

    Panjang baris dari dataset full tanpa anomaly :  (20169, 90)



```python
print('Informasi distribusi tanpa anomaly')
gold_recovery_full_data.describe()
```

    Informasi distribusi tanpa anomaly





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
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>primary_cleaner.input.sulfate</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
      <th>rougher.input.feed.total_concentration</th>
      <th>rougher.output.concentrate.total_concentration</th>
      <th>final.output.concentrate.total_concentration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>...</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
      <td>20169.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.181331</td>
      <td>9.874233</td>
      <td>9.313900</td>
      <td>43.445598</td>
      <td>66.804303</td>
      <td>9.662609</td>
      <td>2.679577</td>
      <td>10.316853</td>
      <td>3.046675</td>
      <td>142.923976</td>
      <td>...</td>
      <td>-462.084519</td>
      <td>15.658728</td>
      <td>-488.753423</td>
      <td>12.212474</td>
      <td>-487.116288</td>
      <td>18.929550</td>
      <td>-505.610047</td>
      <td>56.871675</td>
      <td>69.088854</td>
      <td>56.871675</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.539089</td>
      <td>1.936062</td>
      <td>3.058229</td>
      <td>6.842340</td>
      <td>11.274330</td>
      <td>2.495782</td>
      <td>0.997232</td>
      <td>3.159906</td>
      <td>0.996843</td>
      <td>46.891163</td>
      <td>...</td>
      <td>67.551706</td>
      <td>5.526352</td>
      <td>35.627789</td>
      <td>5.315324</td>
      <td>39.954969</td>
      <td>5.553953</td>
      <td>39.778437</td>
      <td>8.225592</td>
      <td>9.688037</td>
      <td>8.225592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001982</td>
      <td>...</td>
      <td>-800.836914</td>
      <td>-0.372054</td>
      <td>-797.323986</td>
      <td>0.528083</td>
      <td>-800.220337</td>
      <td>-0.079426</td>
      <td>-809.741464</td>
      <td>1.329681</td>
      <td>1.064979</td>
      <td>1.329681</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.242546</td>
      <td>9.101823</td>
      <td>7.554877</td>
      <td>43.190108</td>
      <td>63.167249</td>
      <td>8.045063</td>
      <td>2.004206</td>
      <td>8.692907</td>
      <td>2.453737</td>
      <td>112.364920</td>
      <td>...</td>
      <td>-500.187760</td>
      <td>10.986849</td>
      <td>-500.462260</td>
      <td>8.977016</td>
      <td>-500.130667</td>
      <td>14.982351</td>
      <td>-500.744921</td>
      <td>53.268002</td>
      <td>66.136370</td>
      <td>53.268002</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.084789</td>
      <td>10.082382</td>
      <td>9.153468</td>
      <td>44.895986</td>
      <td>68.146768</td>
      <td>9.788126</td>
      <td>2.732210</td>
      <td>10.494473</td>
      <td>2.992570</td>
      <td>142.500589</td>
      <td>...</td>
      <td>-499.473842</td>
      <td>14.996748</td>
      <td>-499.790931</td>
      <td>11.019896</td>
      <td>-499.933619</td>
      <td>19.953389</td>
      <td>-500.047326</td>
      <td>57.502487</td>
      <td>70.345389</td>
      <td>57.502487</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.934334</td>
      <td>11.031057</td>
      <td>10.939022</td>
      <td>46.213804</td>
      <td>72.730246</td>
      <td>11.207404</td>
      <td>3.337185</td>
      <td>12.011702</td>
      <td>3.588607</td>
      <td>174.911627</td>
      <td>...</td>
      <td>-400.147484</td>
      <td>18.025333</td>
      <td>-498.323325</td>
      <td>14.017375</td>
      <td>-499.416210</td>
      <td>23.989110</td>
      <td>-499.475424</td>
      <td>61.945189</td>
      <td>74.621679</td>
      <td>61.945189</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.001945</td>
      <td>17.031899</td>
      <td>19.615720</td>
      <td>53.611374</td>
      <td>100.000000</td>
      <td>19.552149</td>
      <td>6.086532</td>
      <td>22.861749</td>
      <td>9.789625</td>
      <td>265.983123</td>
      <td>...</td>
      <td>-6.506986</td>
      <td>43.709931</td>
      <td>-244.483566</td>
      <td>27.926001</td>
      <td>-126.463446</td>
      <td>32.188906</td>
      <td>-29.093593</td>
      <td>76.978947</td>
      <td>90.964431</td>
      <td>76.978947</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 89 columns</p>
</div>




```python
print('Informasi distribusi dengan anomaly')
gold_recovery_full.describe()
```

    Informasi distribusi dengan anomaly





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
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>primary_cleaner.input.sulfate</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
      <th>rougher.input.feed.total_concentration</th>
      <th>rougher.output.concentrate.total_concentration</th>
      <th>final.output.concentrate.total_concentration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>...</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
      <td>22716.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.768013</td>
      <td>9.071366</td>
      <td>8.537502</td>
      <td>39.891718</td>
      <td>66.475263</td>
      <td>8.900622</td>
      <td>2.471149</td>
      <td>9.434396</td>
      <td>2.819885</td>
      <td>131.338303</td>
      <td>...</td>
      <td>-477.753153</td>
      <td>14.831127</td>
      <td>-504.279270</td>
      <td>11.588761</td>
      <td>-501.510723</td>
      <td>17.904473</td>
      <td>-520.266675</td>
      <td>51.856810</td>
      <td>61.518427</td>
      <td>51.856810</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.042594</td>
      <td>3.260960</td>
      <td>3.858203</td>
      <td>13.540157</td>
      <td>13.042781</td>
      <td>3.544045</td>
      <td>1.201069</td>
      <td>4.145832</td>
      <td>1.269214</td>
      <td>58.477466</td>
      <td>...</td>
      <td>95.648618</td>
      <td>6.417280</td>
      <td>74.648011</td>
      <td>5.780520</td>
      <td>80.663516</td>
      <td>6.717178</td>
      <td>76.976308</td>
      <td>17.934564</td>
      <td>23.412615</td>
      <td>17.934564</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000003</td>
      <td>...</td>
      <td>-800.836914</td>
      <td>-0.423260</td>
      <td>-799.741097</td>
      <td>0.427084</td>
      <td>-800.258209</td>
      <td>-0.079426</td>
      <td>-810.473526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.011471</td>
      <td>8.737809</td>
      <td>7.036629</td>
      <td>42.353890</td>
      <td>62.258453</td>
      <td>7.669323</td>
      <td>1.780344</td>
      <td>8.050901</td>
      <td>2.297002</td>
      <td>101.198976</td>
      <td>...</td>
      <td>-500.319422</td>
      <td>10.938055</td>
      <td>-500.641708</td>
      <td>8.031211</td>
      <td>-500.171370</td>
      <td>13.031799</td>
      <td>-501.000058</td>
      <td>51.400302</td>
      <td>63.325353</td>
      <td>51.400302</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.949959</td>
      <td>9.910363</td>
      <td>8.858385</td>
      <td>44.639019</td>
      <td>67.981407</td>
      <td>9.477554</td>
      <td>2.643964</td>
      <td>10.174399</td>
      <td>2.910336</td>
      <td>137.084713</td>
      <td>...</td>
      <td>-499.616792</td>
      <td>14.615849</td>
      <td>-499.868380</td>
      <td>10.987789</td>
      <td>-499.953415</td>
      <td>18.002995</td>
      <td>-500.098653</td>
      <td>56.788595</td>
      <td>69.414309</td>
      <td>56.788595</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.857985</td>
      <td>10.927188</td>
      <td>10.667178</td>
      <td>46.106662</td>
      <td>72.941119</td>
      <td>11.084153</td>
      <td>3.282285</td>
      <td>11.840024</td>
      <td>3.552452</td>
      <td>170.993920</td>
      <td>...</td>
      <td>-400.229299</td>
      <td>18.014080</td>
      <td>-498.503626</td>
      <td>13.999903</td>
      <td>-499.499414</td>
      <td>23.007616</td>
      <td>-499.527882</td>
      <td>61.405688</td>
      <td>74.025452</td>
      <td>61.405688</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.001945</td>
      <td>17.031899</td>
      <td>19.615720</td>
      <td>53.611374</td>
      <td>100.000000</td>
      <td>19.552149</td>
      <td>6.086532</td>
      <td>22.861749</td>
      <td>9.789625</td>
      <td>274.409626</td>
      <td>...</td>
      <td>-6.506986</td>
      <td>63.116298</td>
      <td>-244.483566</td>
      <td>39.846228</td>
      <td>-120.190931</td>
      <td>54.876806</td>
      <td>-29.093593</td>
      <td>76.978947</td>
      <td>90.964431</td>
      <td>76.978947</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 89 columns</p>
</div>




```python
# rougher.input.feed pada dataset full tanpa anomali / outlier
gold_recovery_full_data['rougher.input.feed.total_concentration'] =   gold_recovery_full_data[['rougher.output.concentrate_ag',
                                                                                           'rougher.output.concentrate_pb',
                                                                                           'rougher.output.concentrate_sol',
                                                                                           'rougher.output.concentrate_au']].sum(axis=1)
# memriksa nilai anomali
plt.figure(figsize=(10,6))
total_conc_stage_1 = gold_recovery_full_data['rougher.input.feed.total_concentration'] 
plt.hist(total_conc_stage_1, density=True, bins=100)

# menambahkan judul dan nama 
plt.xlabel('rougher.input.feed_total concentration')
plt.ylabel('frequency')
plt.title(" Total Distribusi dari proses konsentrasi pada raw feed tanpa outlier");
```


    
![png](output_61_0.png)
    



```python
# rougher.output.concentrate pada full dataset tanpa anomali / outlier
gold_recovery_full_data['rougher.output.concentrate.total_concentration'] = gold_recovery_full_data[['rougher.output.concentrate_ag',
                                                                                           'rougher.output.concentrate_pb',
                                                                                           'rougher.output.concentrate_sol',
                                                                                           'rougher.output.concentrate_au']].sum(axis=1)

# memriksa nilai anomali
total_conc_stage_2 = gold_recovery_full_data['rougher.output.concentrate.total_concentration']
plt.figure(figsize=(10,6))
plt.hist(total_conc_stage_2, density=True, bins=100)

# menambahkan judul dan nama 
plt.xlabel('rougher.output.concentrate_total concentration')
plt.ylabel('frequency')
plt.title("Total Distribusi dari proses konsentrasi pada rougher concentrate tanpa outlier");
```


    
![png](output_62_0.png)
    



```python
# final.output.concentrate pada full datasettanpa anomali / outlier
gold_recovery_full_data['final.output.concentrate.total_concentration'] = gold_recovery_full_data[['rougher.input.feed_ag',
                                                                                   'rougher.input.feed_pb', 
                                                                                   'rougher.input.feed_sol', 
                                                                                   'rougher.input.feed_au']].sum(axis=1)

# memriksa nilai anomali
total_conc_stage_3 = gold_recovery_full_data['final.output.concentrate.total_concentration'] 
plt.figure(figsize=(10,6))
plt.hist(total_conc_stage_3, density=True, bins=100)

# menambahkan judul dan nama 
plt.xlabel('final.output.concentrate_total concentration')
plt.ylabel('frequency')
plt.title("Total Distribusi dari proses konsentrasi pada final concentrate tanpa anomali ");
```


    
![png](output_63_0.png)
    


Kesimpulan :

Setelah menganalisis distribusi konsentrasi logam di seluruh tahap pemurnian, kita mengamati bahwa konsentrasi emas (au) meningkat dari `rougher.output` ke `final.output`. Hal ini menunjukkan bahwa proses berjalan seperti yang dirancang. lalu dengan membandingkan distribusi ukuran partikel feed untuk dataset train dan test, disini kita mengamati bahwa ukuran rata-rata partikel sama. Contohnya, ukuran partikel feed rata-rata untuk dataset train dan test masing-masing yakni 60 dan 56 pada proses rougher.input.feed_size, dan rata-rata ukuran partikel feed untuk dataset train dan test yakni 7.30 dan 7.2 untuk proses primary_cleaner.input.feed_size. 
kita juga memeriksa anomali dalam konsentrasi total semua zat pada tahap yang berbeda. Kami mengamati beberapa anomali dalam data dengan puncak abnormal dalam konsentrasi total pada 0. Kami menggunakan ambang batas antara 0,95 untuk menghilangkan anomali, serta kita telah  berhasil memfilter anomaly / outlier kita. sekarang kita siap untuk membuat model.

## Membuat model 

Sebelum membuat model dengan menghitung nilai sMAPE, kita akan memfilter dataset full tanpa outlier ke dataset train dan test.


```python
# membuat index `date` untuk memfilter dataset train dan test
gold_recovery_test.set_index('date', inplace=True)
gold_recovery_train.set_index('date', inplace=True)
gold_recovery_full_data.set_index('date', inplace=True)
```


```python
# memfilter index dari dataset train berdasarkan full dataset tanpa outlier
gold_recovery_train_data_index = gold_recovery_full_data.index.intersection(gold_recovery_train.index)
print('Panjang baris dari dataset train tanpa anomaly : ', format(gold_recovery_train_data_index.shape))
```

    Panjang baris dari dataset train tanpa anomaly :  (14796,)



```python
# train dataset baru 
gold_recovery_train_data = gold_recovery_full_data.loc[gold_recovery_train_data_index]
gold_recovery_train_data.reset_index(inplace=True)
gold_recovery_train_data
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
      <th>date</th>
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
      <th>rougher.input.feed.total_concentration</th>
      <th>rougher.output.concentrate.total_concentration</th>
      <th>final.output.concentrate.total_concentration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-15 00:00:00</td>
      <td>6.055403</td>
      <td>9.889648</td>
      <td>5.507324</td>
      <td>42.192020</td>
      <td>70.541216</td>
      <td>10.411962</td>
      <td>0.895447</td>
      <td>16.904297</td>
      <td>2.143149</td>
      <td>...</td>
      <td>-504.715942</td>
      <td>9.925633</td>
      <td>-498.310211</td>
      <td>8.079666</td>
      <td>-500.470978</td>
      <td>14.151341</td>
      <td>-605.841980</td>
      <td>66.424950</td>
      <td>66.424950</td>
      <td>51.680034</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-15 01:00:00</td>
      <td>6.029369</td>
      <td>9.968944</td>
      <td>5.257781</td>
      <td>42.701629</td>
      <td>69.266198</td>
      <td>10.462676</td>
      <td>0.927452</td>
      <td>16.634514</td>
      <td>2.224930</td>
      <td>...</td>
      <td>-501.331529</td>
      <td>10.039245</td>
      <td>-500.169983</td>
      <td>7.984757</td>
      <td>-500.582168</td>
      <td>13.998353</td>
      <td>-599.787184</td>
      <td>67.012710</td>
      <td>67.012710</td>
      <td>50.659114</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-15 02:00:00</td>
      <td>6.055926</td>
      <td>10.213995</td>
      <td>5.383759</td>
      <td>42.657501</td>
      <td>68.116445</td>
      <td>10.507046</td>
      <td>0.953716</td>
      <td>16.208849</td>
      <td>2.257889</td>
      <td>...</td>
      <td>-501.133383</td>
      <td>10.070913</td>
      <td>-500.129135</td>
      <td>8.013877</td>
      <td>-500.517572</td>
      <td>14.028663</td>
      <td>-601.427363</td>
      <td>66.103793</td>
      <td>66.103793</td>
      <td>50.609929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-15 03:00:00</td>
      <td>6.047977</td>
      <td>9.977019</td>
      <td>4.858634</td>
      <td>42.689819</td>
      <td>68.347543</td>
      <td>10.422762</td>
      <td>0.883763</td>
      <td>16.532835</td>
      <td>2.146849</td>
      <td>...</td>
      <td>-501.193686</td>
      <td>9.970366</td>
      <td>-499.201640</td>
      <td>7.977324</td>
      <td>-500.255908</td>
      <td>14.005551</td>
      <td>-599.996129</td>
      <td>65.752751</td>
      <td>65.752751</td>
      <td>51.061546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-15 04:00:00</td>
      <td>6.148599</td>
      <td>10.142511</td>
      <td>4.939416</td>
      <td>42.774141</td>
      <td>66.927016</td>
      <td>10.360302</td>
      <td>0.792826</td>
      <td>16.525686</td>
      <td>2.055292</td>
      <td>...</td>
      <td>-501.053894</td>
      <td>9.925709</td>
      <td>-501.686727</td>
      <td>7.894242</td>
      <td>-500.356035</td>
      <td>13.996647</td>
      <td>-601.496691</td>
      <td>65.908382</td>
      <td>65.908382</td>
      <td>47.859163</td>
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
      <th>14791</th>
      <td>2018-08-18 06:59:59</td>
      <td>3.224920</td>
      <td>11.356233</td>
      <td>6.803482</td>
      <td>46.713954</td>
      <td>73.755150</td>
      <td>8.769645</td>
      <td>3.141541</td>
      <td>10.403181</td>
      <td>1.529220</td>
      <td>...</td>
      <td>-499.740028</td>
      <td>18.006038</td>
      <td>-499.834374</td>
      <td>13.001114</td>
      <td>-500.155694</td>
      <td>20.007840</td>
      <td>-501.296428</td>
      <td>70.781325</td>
      <td>70.781325</td>
      <td>53.415050</td>
    </tr>
    <tr>
      <th>14792</th>
      <td>2018-08-18 07:59:59</td>
      <td>3.195978</td>
      <td>11.349355</td>
      <td>6.862249</td>
      <td>46.866780</td>
      <td>69.049291</td>
      <td>8.897321</td>
      <td>3.130493</td>
      <td>10.549470</td>
      <td>1.612542</td>
      <td>...</td>
      <td>-500.251357</td>
      <td>17.998535</td>
      <td>-500.395178</td>
      <td>12.954048</td>
      <td>-499.895163</td>
      <td>19.968498</td>
      <td>-501.041608</td>
      <td>70.539603</td>
      <td>70.539603</td>
      <td>53.696482</td>
    </tr>
    <tr>
      <th>14793</th>
      <td>2018-08-18 08:59:59</td>
      <td>3.109998</td>
      <td>11.434366</td>
      <td>6.886013</td>
      <td>46.795691</td>
      <td>67.002189</td>
      <td>8.529606</td>
      <td>2.911418</td>
      <td>11.115147</td>
      <td>1.596616</td>
      <td>...</td>
      <td>-499.857027</td>
      <td>18.019543</td>
      <td>-500.451156</td>
      <td>13.023431</td>
      <td>-499.914391</td>
      <td>19.990885</td>
      <td>-501.518452</td>
      <td>55.376330</td>
      <td>55.376330</td>
      <td>54.589604</td>
    </tr>
    <tr>
      <th>14794</th>
      <td>2018-08-18 09:59:59</td>
      <td>3.367241</td>
      <td>11.625587</td>
      <td>6.799433</td>
      <td>46.408188</td>
      <td>65.523246</td>
      <td>8.777171</td>
      <td>2.819214</td>
      <td>10.463847</td>
      <td>1.602879</td>
      <td>...</td>
      <td>-500.314711</td>
      <td>17.979515</td>
      <td>-499.272871</td>
      <td>12.992404</td>
      <td>-499.976268</td>
      <td>20.013986</td>
      <td>-500.625471</td>
      <td>69.201689</td>
      <td>69.201689</td>
      <td>54.027355</td>
    </tr>
    <tr>
      <th>14795</th>
      <td>2018-08-18 10:59:59</td>
      <td>3.598375</td>
      <td>11.737832</td>
      <td>6.717509</td>
      <td>46.299438</td>
      <td>70.281454</td>
      <td>8.406690</td>
      <td>2.517518</td>
      <td>10.652193</td>
      <td>1.389434</td>
      <td>...</td>
      <td>-500.220296</td>
      <td>17.963512</td>
      <td>-499.939490</td>
      <td>12.990306</td>
      <td>-500.080993</td>
      <td>19.990336</td>
      <td>-499.191575</td>
      <td>69.544003</td>
      <td>69.544003</td>
      <td>53.535054</td>
    </tr>
  </tbody>
</table>
<p>14796 rows × 90 columns</p>
</div>




```python
# memfilter index dari dataset test berdasarkan full dataset tanpa outlier
gold_recovery_test_data_index = gold_recovery_full_data.index.intersection(gold_recovery_test.index)
gold_recovery_test_data_index.shape
```




    (5373,)




```python
# test dataset baru
gold_recovery_test_data = gold_recovery_full_data.loc[gold_recovery_test_data_index]
gold_recovery_test_data.reset_index(inplace=True)
gold_recovery_test_data
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
      <th>date</th>
      <th>final.output.concentrate_ag</th>
      <th>final.output.concentrate_pb</th>
      <th>final.output.concentrate_sol</th>
      <th>final.output.concentrate_au</th>
      <th>final.output.recovery</th>
      <th>final.output.tail_ag</th>
      <th>final.output.tail_pb</th>
      <th>final.output.tail_sol</th>
      <th>final.output.tail_au</th>
      <th>...</th>
      <th>secondary_cleaner.state.floatbank4_b_level</th>
      <th>secondary_cleaner.state.floatbank5_a_air</th>
      <th>secondary_cleaner.state.floatbank5_a_level</th>
      <th>secondary_cleaner.state.floatbank5_b_air</th>
      <th>secondary_cleaner.state.floatbank5_b_level</th>
      <th>secondary_cleaner.state.floatbank6_a_air</th>
      <th>secondary_cleaner.state.floatbank6_a_level</th>
      <th>rougher.input.feed.total_concentration</th>
      <th>rougher.output.concentrate.total_concentration</th>
      <th>final.output.concentrate.total_concentration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-09-01 00:59:59</td>
      <td>7.578381</td>
      <td>10.466295</td>
      <td>11.990938</td>
      <td>40.743891</td>
      <td>70.273583</td>
      <td>12.688885</td>
      <td>3.844413</td>
      <td>11.075686</td>
      <td>4.537988</td>
      <td>...</td>
      <td>-501.289139</td>
      <td>7.946562</td>
      <td>-432.317850</td>
      <td>4.872511</td>
      <td>-500.037437</td>
      <td>26.705889</td>
      <td>-499.709414</td>
      <td>79.939838</td>
      <td>79.939838</td>
      <td>72.871822</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-09-01 01:59:59</td>
      <td>7.813838</td>
      <td>10.581152</td>
      <td>12.216172</td>
      <td>39.604292</td>
      <td>68.910432</td>
      <td>12.829171</td>
      <td>3.918901</td>
      <td>11.132824</td>
      <td>4.675117</td>
      <td>...</td>
      <td>-499.634209</td>
      <td>7.958270</td>
      <td>-525.839648</td>
      <td>4.878850</td>
      <td>-500.162375</td>
      <td>25.019940</td>
      <td>-499.819438</td>
      <td>81.118880</td>
      <td>81.118880</td>
      <td>71.669225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-09-01 02:59:59</td>
      <td>7.623392</td>
      <td>10.424024</td>
      <td>12.313710</td>
      <td>40.724190</td>
      <td>68.143213</td>
      <td>12.977846</td>
      <td>4.026561</td>
      <td>10.990134</td>
      <td>4.828907</td>
      <td>...</td>
      <td>-500.827423</td>
      <td>8.071056</td>
      <td>-500.801673</td>
      <td>4.905125</td>
      <td>-499.828510</td>
      <td>24.994862</td>
      <td>-500.622559</td>
      <td>79.267044</td>
      <td>79.267044</td>
      <td>73.202598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-09-01 03:59:59</td>
      <td>8.552457</td>
      <td>10.503229</td>
      <td>13.074570</td>
      <td>39.290997</td>
      <td>67.776393</td>
      <td>12.451947</td>
      <td>3.780702</td>
      <td>11.155935</td>
      <td>4.969620</td>
      <td>...</td>
      <td>-499.474407</td>
      <td>7.897085</td>
      <td>-500.868509</td>
      <td>4.931400</td>
      <td>-499.963623</td>
      <td>24.948919</td>
      <td>-498.709987</td>
      <td>81.335254</td>
      <td>81.335254</td>
      <td>70.757057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-09-01 04:59:59</td>
      <td>8.078781</td>
      <td>10.222788</td>
      <td>12.475427</td>
      <td>40.254524</td>
      <td>61.467078</td>
      <td>11.827846</td>
      <td>3.632272</td>
      <td>11.403663</td>
      <td>5.256806</td>
      <td>...</td>
      <td>-500.397500</td>
      <td>8.107890</td>
      <td>-509.526725</td>
      <td>4.957674</td>
      <td>-500.360026</td>
      <td>25.003331</td>
      <td>-500.856333</td>
      <td>80.902631</td>
      <td>80.902631</td>
      <td>68.654396</td>
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
      <th>5368</th>
      <td>2017-12-31 19:59:59</td>
      <td>5.000174</td>
      <td>9.710255</td>
      <td>10.845459</td>
      <td>46.400415</td>
      <td>68.919891</td>
      <td>13.944836</td>
      <td>3.373224</td>
      <td>13.766506</td>
      <td>3.890235</td>
      <td>...</td>
      <td>-499.673279</td>
      <td>7.977259</td>
      <td>-499.516126</td>
      <td>5.933319</td>
      <td>-499.965973</td>
      <td>8.987171</td>
      <td>-499.755909</td>
      <td>67.518199</td>
      <td>67.518199</td>
      <td>68.910849</td>
    </tr>
    <tr>
      <th>5369</th>
      <td>2017-12-31 20:59:59</td>
      <td>4.956679</td>
      <td>9.727962</td>
      <td>9.705617</td>
      <td>46.657393</td>
      <td>68.440582</td>
      <td>12.624143</td>
      <td>2.974607</td>
      <td>14.177795</td>
      <td>3.809054</td>
      <td>...</td>
      <td>-499.122723</td>
      <td>9.288553</td>
      <td>-496.892967</td>
      <td>7.372897</td>
      <td>-499.942956</td>
      <td>8.986832</td>
      <td>-499.903761</td>
      <td>67.695694</td>
      <td>67.695694</td>
      <td>67.166899</td>
    </tr>
    <tr>
      <th>5370</th>
      <td>2017-12-31 21:59:59</td>
      <td>4.779534</td>
      <td>9.818943</td>
      <td>8.255551</td>
      <td>47.337296</td>
      <td>67.092759</td>
      <td>12.134647</td>
      <td>2.843604</td>
      <td>13.219960</td>
      <td>3.909903</td>
      <td>...</td>
      <td>-499.936252</td>
      <td>10.989181</td>
      <td>-498.347898</td>
      <td>9.020944</td>
      <td>-500.040448</td>
      <td>8.982038</td>
      <td>-497.789882</td>
      <td>72.437678</td>
      <td>72.437678</td>
      <td>66.955814</td>
    </tr>
    <tr>
      <th>5371</th>
      <td>2017-12-31 22:59:59</td>
      <td>4.472036</td>
      <td>9.473869</td>
      <td>8.466341</td>
      <td>48.258531</td>
      <td>68.061186</td>
      <td>12.331412</td>
      <td>2.889243</td>
      <td>12.165999</td>
      <td>3.749126</td>
      <td>...</td>
      <td>-499.723143</td>
      <td>11.011607</td>
      <td>-499.985046</td>
      <td>9.009783</td>
      <td>-499.937902</td>
      <td>9.012660</td>
      <td>-500.154284</td>
      <td>73.701926</td>
      <td>73.701926</td>
      <td>67.420094</td>
    </tr>
    <tr>
      <th>5372</th>
      <td>2017-12-31 23:59:59</td>
      <td>4.604069</td>
      <td>9.413112</td>
      <td>8.825604</td>
      <td>47.714922</td>
      <td>71.699976</td>
      <td>12.149829</td>
      <td>2.559168</td>
      <td>12.594390</td>
      <td>3.212437</td>
      <td>...</td>
      <td>-499.948518</td>
      <td>10.986607</td>
      <td>-500.658027</td>
      <td>8.989497</td>
      <td>-500.337588</td>
      <td>8.988632</td>
      <td>-500.764937</td>
      <td>73.496364</td>
      <td>73.496364</td>
      <td>65.583618</td>
    </tr>
  </tbody>
</table>
<p>5373 rows × 90 columns</p>
</div>



Dataset telah berhasil di filter dengan menghilangkan outliernya

Pada tahap membuat model kita akan memprediksi 2 values yakni :

1. rougher concentrate recovery `ougher.output.recovery`
2. final concentrate recovery `final.output.recovery`

Untuk melakukan itu, kita akan membuat dan melatih berbagai model, dan mengevaluasinya menggunakan validasi silang. Kita menggunakan rumus sMAPE dan Final sMAPE sebagai metrik evaluasi. Rumus sMAPE yakni:


```python
from IPython.display import Image
Image(url= 'https://pictures.s3.yandex.net/resources/smape_1576239058_1589899769.jpg',width = 500, height = 200 )
```




<img src="https://pictures.s3.yandex.net/resources/smape_1576239058_1589899769.jpg" width="500" height="200"/>




```python
# Fungsi untuk menghitung sMAPE
def smape(y_true, y_pred):
    smape = (np.abs(y_true - y_pred)/((np.abs(y_true) + np.abs(y_pred))/2)).mean()
    return smape

# fungsi untuk menghitung final sMAPE
def smape_final(y_true, y_pred):
    smape_out_rougher = smape(y_true[:,0], y_pred[:,0])
    smape_out_final = smape(y_true[:,1], y_pred[:,1])
    return ((0.25 * smape_out_rougher) + (0.75 * smape_out_final))

# sMAPE final Score
smape_score = make_scorer(smape_final)
```


```python
# Mendefenisiskan variables features dan target

features_train = gold_recovery_train_data.drop(['date', 'rougher.output.recovery', 'final.output.recovery'], axis=1)
features_test = gold_recovery_test_data.drop(['date', 'rougher.output.recovery', 'final.output.recovery'], axis=1)
target_train = gold_recovery_train_data[['rougher.output.recovery', 'final.output.recovery']]
target_test = gold_recovery_test_data[['rougher.output.recovery', 'final.output.recovery']]
```


```python
# # menstandarkan data numerik dengan features scaling 
scaler = StandardScaler()
scaler.fit(features_train)

# mengubah train set dan test set menggunggunakan transform()
features_train = scaler.transform(features_train)
features_test  = scaler.transform(features_test)

# mengubah target data menjadi numpy array
target_train = target_train.values
target_test = target_test.values
```

# Baseline model


```python
# baseline model menggunakan dummy regressor
start_time = timeit.default_timer()
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(features_train, target_train)
dummy_regr_test_predictions = dummy_regr.predict(features_test)

# cross-validation untuk dumy regressor
cv_score_dm = cross_val_score(dummy_regr, features_train, target_train, cv=5, scoring=smape_score)
print('Rata-rata sMAPE     : {:.2%}'.format(cv_score_dm.mean()))
print()
print('rentang nilai sMAPE :', cv_score_dm)
print()
print('Waktu Pelaksanaan   : ' + str((timeit.default_timer() - start_time)) + ' mins')
print()

#evaluasi model dengan metrik sMAPE
print('Skor final sMAPE : {:.2%}'.format(smape_final(target_test, dummy_regr_test_predictions)))
```

    Rata-rata sMAPE     : 11.49%
    
    rentang nilai sMAPE : [0.1194492  0.10572608 0.10316369 0.12331096 0.12301836]
    
    Waktu Pelaksanaan   : 0.05075993435457349 mins
    
    Skor final sMAPE : 10.30%


# Linear Regression Model


```python
# Membuat Model 
start_time = timeit.default_timer()
lr_regr = LinearRegression().fit(features_train, target_train) # melatih model 
lr_regr_test_predictions = lr_regr.predict(features_test)

# cross-validation untuk linear Regressoion model
cv_score_lr = cross_val_score(lr_regr, features_train, target_train, cv=5, scoring=smape_score)
print('Rata-rata sMAPE     : {:.2%}'.format(cv_score_lr.mean()))
print()
print('rentang nilai sMAPE :', cv_score_lr)
print()
print('Waktu Pelaksanaan   : ' + str((timeit.default_timer() - start_time)) + ' mins')
print()

#evaluasi model dengan metrik sMAPE
print('Skor final sMAPE    : {:.2%}'.format(smape_final(target_test, lr_regr_test_predictions)))
```

    Rata-rata sMAPE     : 6.51%
    
    rentang nilai sMAPE : [0.06982115 0.06148266 0.06417689 0.05731111 0.07261624]
    
    Waktu Pelaksanaan   : 1.1915411693044007 mins
    
    Skor final sMAPE    : 5.87%


# Decision Tree Regression model


```python
# Membuat Model 
start_time = timeit.default_timer()
dt_regr = DecisionTreeRegressor().fit(features_train, target_train)  # melatih model
dt_regr_test_predictions = dt_regr.predict(features_test)

# cross-validation for decision tree regression
cv_score_dt = cross_val_score(dt_regr, features_train, target_train, cv=5, scoring=smape_score)
print('Rata-rata sMAPE     :  {:.4%}'.format(cv_score_dt.mean()))
print()
print('rentang nilai sMAPE : ', cv_score_dt)
print()
print("Waktu Pelaksanaan   : " + str((timeit.default_timer() - start_time)/60) + ' mins')
print()
#evaluasi model dengan metrik sMAPE
print('Skor final sMAPE    : {:.2%}'.format(smape_final(target_test, dt_regr_test_predictions)))
```

    Rata-rata sMAPE     :  nan%
    
    rentang nilai sMAPE :  [      nan       nan       nan       nan 0.0667237]
    
    Waktu Pelaksanaan   : 0.22992946635155628 mins
    
    Skor final sMAPE    : nan%



```python
# Menentukan skema cross validation 
for i in [2, 4, 8, 16]:
    dt = DecisionTreeRegressor(max_depth = i, random_state = 12345)
    dt.fit(features_train, target_train)
    cross_val_scores_dt = cross_val_score(dt, features_train, target_train, cv=5, scoring=smape_score)
    print('Max depth: ' + str(i) + ', rata-rata smape: {:.4%}'.format(cross_val_scores_dt.mean()))
```

    Max depth: 2, rata-rata smape: 10.4660%
    Max depth: 4, rata-rata smape: nan%
    Max depth: 8, rata-rata smape: nan%
    Max depth: 16, rata-rata smape: nan%



```python
# Optimasi hyperparameters
start_time = timeit.default_timer()

# Menentukan hyperparameters untuk di tuning
params_ = {
    "max_depth" : [2, 4, 8, 16],
    "min_samples_split" : [2, 4, 8, 16]
    }
# Menentukan model
dt_regressor = DecisionTreeRegressor()

# Menentukan grid search
grid_search_dt = GridSearchCV(estimator = dt_regressor, param_grid = params_, scoring=smape_score)

# menjalankan model
grid_search_dt.fit(features_train, target_train)

# Hasil hyperparameters terbaik
print('Hasil Hyperparameters terbaik : {}'.format(grid_search_dt.best_params_))
print()
print("Waktu Pelaksanaan             : " + str((timeit.default_timer() - start_time)/60) + ' mins')
```


```python
# Membuat fungsi untuk model decision tree regressor
def decision_tree_regressor(X_train, y_train, X_test, y_test):

    # membuat model 
    dt_model = DecisionTreeRegressor(**grid_search_dt.best_params_)
    dt_model.fit(X_train, y_train) # train the model 
   
    # membuat prediksi pada test dataset
    dt_test_predictions = dt_model.predict(X_test)
    
    # Menghitung final sMAPE decision tree regressor
    print('Skor final sMAPE  : {:.2%}'.format(smape_final(target_test, dt_test_predictions)))

```


```python
# Menjalankan fungsi
decision_tree_regressor(features_train, target_train, features_test, target_test)
```

## Kesimpulan Akhir

Pada tahap ini, kita memeriksa data dan mengamati bahwa : 
1. data gold_recovery_train, memiliki 16.860 rows, 86 features, dan 85 columns ngan NA. 
2. data gold_recovery_test , memiliki 5.856 rows, 52 features, dan 51 columns dengan NA. 
3. data gold_recovery_full , memiliki 22.716 baris, 86 features, dan 85 columns dengan NA. 

lalu setelah memeriksa dataset kita mendapati bahwa features 'date' memiliki tipe data yang tidak tepat. setelah merubah tipe data ke tipe data datetime kita mengisi nilai yang hilang dengan metode ffill metode ini mengganti nilai yang hilang dengan nilai dari baris sebelumnya (atau kolom sebelumnya, jika parameter sumbu diatur ke 'kolom').

---

Dari perhitungan yang dilakukan, kita dapat melihat bahwa  calculated_recovery dan rougher.output.recovery memiliki nilai yang sama. dan score Mean Absolute Erronya adalah 0,0. Ini menunjukkan bahwa nilai yang dihitung dari proses pemulihan yang disimulasikan mirip dengan rougher.output.recovery.

---

Setelah memeriksa fitur yang tidak tersedia di dataset test, dapat dilihat dataset train memiliki 34 fitur  yang tidak tersedia di dataset test. Fitur yang tidak terdapat pada dataset test antara lain fitur yang mengandung konsentrasi logam Au = emas, Ag = perak , Pb = timbal. dan jenis parameternya adalah output — product parameters, calculation — calculation characteristics


---


Kami menjumlahkan konsentrasi total semua zat pada tahapan yang berbeda dalam proses yaitu. raw feed `[rougher.input.feed.total_concentration]`, rougher concetrate `[rougher.output.concentrate.total_concentration]`, dan final concentrate  `[final.output.concentrate.total_concentration]`. lalu dengan memplot histogram dan mengamati puncak di ketiga plot menunjukkan anomali dalam data. Sangat aneh jika memiliki konsentrasi total sekitar 0, sehingga puncaknya adalah nilai abnormal dan Salah satu alasan yang mungkin untuk anomali tersebut adalah kesalahan entri data atau mewakili pengamatan yang terjadi dalam kondisi yang tidak biasa. Atau, mungkin itu adalah oservasiyang mungkin secara akurat menggambarkan variabilitas di wilayah studi. kita tidak pernah tahu tetapi satu hal yang harus dilakukan adalah menghapus anomali ini agar tidak menimbulkan bias, dengan menghilangkannya dan menetapkan ambang batas 0,95. 

---

Setelah membuat prediksi dengan menggunakan rumus sMAPE, final sMAPE dan mengevaluasinya menggunakan validasi silang, kita mendapatkan hasil sebagai berikut :

1. Dummy Regressor dengan Skor final sMAPE                         : 10.30%
2. Linear Regression dengan Skor final sMAPE                       : 5.87%
3. Decision Tree Regressor tanpa hyperparameters Skor final sMAPE  : nan%
4. Decision Tree Regressor tuning hyperparameters Skor final sMAPE : 9.08% 
