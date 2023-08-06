# Deskripsi Proyek

Nasabah Bank Beta pergi meninggalkan perusahaan: sedikit demi sedikit, jumlah mereka berkurang setiap bulannya. Para pegawai bank menyadari bahwa lebih murah untuk mempertahankan nasabah lama mereka yang setia daripada menarik nasabah baru.
Pada kasus ini, tugas kita adalah untuk memprediksi apakah seorang nasabah akan segera meninggalkan bank atau tidak. 
Anda memiliki data terkait perilaku para klien di masa lalu dan riwayat pemutusan kontrak mereka dengan bank.
Buatlah sebuah model dengan skor F1 semaksimal mungkin. Untuk bisa dinyatakan lulus dari peninjauan, Anda memerlukan skor F1 minimal 0,59 untuk test dataset. Periksa nilai F1 untuk test set.
Selain itu, ukur metrik AUC-ROC dan bandingkan metrik tersebut dengan skor F1.
_______________
Deskripsi Data :
Data yang Anda butuhkan bisa ditemukan di file /datasets/Churn.csv. Unduh dataset.

Fitur-fitur :
1. RowNumber — indeks string data
2. CustomerId — ID pelanggan
3. Surname — nama belakang
4. CreditScore — skor kredit
5. Geography — negara domisili
6. Gender — gender
7. Age — umur
8. Tenure — jangka waktu jatuh tempo untuk deposito tetap nasabah (tahun)
9. Balance — saldo rekening
10. NumOfProducts — jumlah produk bank yang digunakan oleh nasabah
11. HasCrCard — apakah nasabah memiliki kartu kredit
12. IsActiveMember — tingkat keaktifan nasabah
13. EstimatedSalary — estimasi gaji

Target
________
1. Exited — apakah nasabah telah berhenti

## Mengunduh dan mempersiapkan datanya

# Memuat Libary yang dibutuhkan untuk pemrosesan data


```python
# import pandas and numpy untuk proses dan manipulasi data
import pandas as pd
import numpy as np 
import random

# Import seaborn untuk statistika data visualisasi
import seaborn as sns

# import matplotlib untuk data visualisasi
import matplotlib.pyplot as plt 
%matplotlib inline
plt.rcParams.update({'figure.max_open_warning': 0})

# import train_test_split untuk membagi data
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler 
pd.options.mode.chained_assignment = None #menghilangkan notif CopyWarning


# import modul machine learning dari library sklearn
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression 
from catboost import CatBoostClassifier


# import sanity check untuk memeriksa fungsi terhadap model
from sklearn.metrics import *
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, roc_curve)

# import sklearn utilities
from sklearn.utils import shuffle

# import warnings untuk menghapus peringatan saat dataset di manipulasi

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
```

# Memuat Data dari csv agar dapat dijalankan dengan pandas untuk menjadi DataFrame


```python
df = pd.read_csv('https://code.s3.yandex.net/datasets/Churn.csv')
```

# Memuat Informasi dari dataset dan mempelajari dataset


```python
print('Tabel dari dataset')
df.sample(10)
```

    Tabel dari dataset





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
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4792</th>
      <td>4793</td>
      <td>15809991</td>
      <td>Ferrari</td>
      <td>756</td>
      <td>Spain</td>
      <td>Male</td>
      <td>19</td>
      <td>NaN</td>
      <td>130274.22</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>133535.29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2106</th>
      <td>2107</td>
      <td>15659931</td>
      <td>Ibezimako</td>
      <td>637</td>
      <td>Germany</td>
      <td>Female</td>
      <td>55</td>
      <td>1.0</td>
      <td>123378.20</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>81431.99</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9776</th>
      <td>9777</td>
      <td>15700714</td>
      <td>Hollis</td>
      <td>747</td>
      <td>France</td>
      <td>Male</td>
      <td>29</td>
      <td>7.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>141706.43</td>
      <td>0</td>
    </tr>
    <tr>
      <th>311</th>
      <td>312</td>
      <td>15702919</td>
      <td>Collins</td>
      <td>729</td>
      <td>Germany</td>
      <td>Male</td>
      <td>30</td>
      <td>6.0</td>
      <td>63669.42</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>145111.37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>931</th>
      <td>932</td>
      <td>15700476</td>
      <td>Azubuike</td>
      <td>564</td>
      <td>Germany</td>
      <td>Male</td>
      <td>41</td>
      <td>NaN</td>
      <td>103522.75</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>34338.21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>486</th>
      <td>487</td>
      <td>15758639</td>
      <td>Moran</td>
      <td>641</td>
      <td>France</td>
      <td>Male</td>
      <td>37</td>
      <td>7.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>75248.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4478</th>
      <td>4479</td>
      <td>15622443</td>
      <td>Marshall</td>
      <td>549</td>
      <td>France</td>
      <td>Male</td>
      <td>31</td>
      <td>4.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>25684.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8479</th>
      <td>8480</td>
      <td>15807568</td>
      <td>Wright</td>
      <td>632</td>
      <td>France</td>
      <td>Male</td>
      <td>50</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>57942.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6269</th>
      <td>6270</td>
      <td>15734626</td>
      <td>Gibson</td>
      <td>652</td>
      <td>Spain</td>
      <td>Female</td>
      <td>36</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>19302.78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1556</th>
      <td>1557</td>
      <td>15772777</td>
      <td>Onyemachukwu</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>29</td>
      <td>10.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>94815.04</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Informasi keseluruhan dari dataset :')
df.info()
```

    Informasi keseluruhan dari dataset :
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   RowNumber        10000 non-null  int64  
     1   CustomerId       10000 non-null  int64  
     2   Surname          10000 non-null  object 
     3   CreditScore      10000 non-null  int64  
     4   Geography        10000 non-null  object 
     5   Gender           10000 non-null  object 
     6   Age              10000 non-null  int64  
     7   Tenure           9091 non-null   float64
     8   Balance          10000 non-null  float64
     9   NumOfProducts    10000 non-null  int64  
     10  HasCrCard        10000 non-null  int64  
     11  IsActiveMember   10000 non-null  int64  
     12  EstimatedSalary  10000 non-null  float64
     13  Exited           10000 non-null  int64  
    dtypes: float64(3), int64(8), object(3)
    memory usage: 1.1+ MB



```python
print('panjang baris dari dataset :')
df.shape
```

    panjang baris dari dataset :





    (10000, 14)




```python
print('memeriksa nilai yang hilang / na pada dataset :')
df.isna().sum()
```

    memeriksa nilai yang hilang / na pada dataset :





    RowNumber            0
    CustomerId           0
    Surname              0
    CreditScore          0
    Geography            0
    Gender               0
    Age                  0
    Tenure             909
    Balance              0
    NumOfProducts        0
    HasCrCard            0
    IsActiveMember       0
    EstimatedSalary      0
    Exited               0
    dtype: int64




```python
print('persentase nilai yang hilang pada dataset Tenure')
tenur_null = df['Tenure'].isna().sum()
tenur_per = (df['Tenure'].isna().sum() / df.shape[0])
print('Persentase : {:.2f} %, dari nilai keseluruhan yang hilang : {}'.format(tenur_per, tenur_null))

```

    persentase nilai yang hilang pada dataset Tenure
    Persentase : 0.09 %, dari nilai keseluruhan yang hilang : 909



```python
# Data yang hilang pada columns 'Tenure'
df[df['Tenure'].isna()]
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
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>15589475</td>
      <td>Azikiwe</td>
      <td>591</td>
      <td>Spain</td>
      <td>Female</td>
      <td>39</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>140469.38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>15766205</td>
      <td>Yin</td>
      <td>550</td>
      <td>Germany</td>
      <td>Male</td>
      <td>38</td>
      <td>NaN</td>
      <td>103391.38</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>90878.13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>52</td>
      <td>15768193</td>
      <td>Trevisani</td>
      <td>585</td>
      <td>Germany</td>
      <td>Male</td>
      <td>36</td>
      <td>NaN</td>
      <td>146050.97</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>86424.57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>54</td>
      <td>15702298</td>
      <td>Parkhill</td>
      <td>655</td>
      <td>Germany</td>
      <td>Male</td>
      <td>41</td>
      <td>NaN</td>
      <td>125561.97</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>164040.94</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>61</td>
      <td>15651280</td>
      <td>Hunter</td>
      <td>742</td>
      <td>Germany</td>
      <td>Male</td>
      <td>35</td>
      <td>NaN</td>
      <td>136857.00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>84509.57</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>9944</th>
      <td>9945</td>
      <td>15703923</td>
      <td>Cameron</td>
      <td>744</td>
      <td>Germany</td>
      <td>Male</td>
      <td>41</td>
      <td>NaN</td>
      <td>190409.34</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>138361.48</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9956</th>
      <td>9957</td>
      <td>15707861</td>
      <td>Nucci</td>
      <td>520</td>
      <td>France</td>
      <td>Female</td>
      <td>46</td>
      <td>NaN</td>
      <td>85216.61</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>117369.52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9964</th>
      <td>9965</td>
      <td>15642785</td>
      <td>Douglas</td>
      <td>479</td>
      <td>France</td>
      <td>Male</td>
      <td>34</td>
      <td>NaN</td>
      <td>117593.48</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>113308.29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>9986</td>
      <td>15586914</td>
      <td>Nepean</td>
      <td>659</td>
      <td>France</td>
      <td>Male</td>
      <td>36</td>
      <td>NaN</td>
      <td>123841.49</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96833.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>10000</td>
      <td>15628319</td>
      <td>Walker</td>
      <td>792</td>
      <td>France</td>
      <td>Female</td>
      <td>28</td>
      <td>NaN</td>
      <td>130142.79</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38190.78</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>909 rows × 14 columns</p>
</div>




```python
print('memeriksa duplikat pada dataset')
df.duplicated().sum()
```

    memeriksa duplikat pada dataset





    0




```python
print('distribusi statisik tipe data numerik pada dataset')
df.describe()
```

    distribusi statisik tipe data numerik pada dataset





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
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.00000</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>9091.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5000.50000</td>
      <td>1.569094e+07</td>
      <td>650.528800</td>
      <td>38.921800</td>
      <td>4.997690</td>
      <td>76485.889288</td>
      <td>1.530200</td>
      <td>0.70550</td>
      <td>0.515100</td>
      <td>100090.239881</td>
      <td>0.203700</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2886.89568</td>
      <td>7.193619e+04</td>
      <td>96.653299</td>
      <td>10.487806</td>
      <td>2.894723</td>
      <td>62397.405202</td>
      <td>0.581654</td>
      <td>0.45584</td>
      <td>0.499797</td>
      <td>57510.492818</td>
      <td>0.402769</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.556570e+07</td>
      <td>350.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>11.580000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2500.75000</td>
      <td>1.562853e+07</td>
      <td>584.000000</td>
      <td>32.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>51002.110000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5000.50000</td>
      <td>1.569074e+07</td>
      <td>652.000000</td>
      <td>37.000000</td>
      <td>5.000000</td>
      <td>97198.540000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>100193.915000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7500.25000</td>
      <td>1.575323e+07</td>
      <td>718.000000</td>
      <td>44.000000</td>
      <td>7.000000</td>
      <td>127644.240000</td>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>149388.247500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10000.00000</td>
      <td>1.581569e+07</td>
      <td>850.000000</td>
      <td>92.000000</td>
      <td>10.000000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>199992.480000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('distribusi statisik tipe data kategorik pada dataset')
df.describe(include='object')
```

    distribusi statisik tipe data kategorik pada dataset





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
      <th>Surname</th>
      <th>Geography</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2932</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Smith</td>
      <td>France</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>32</td>
      <td>5014</td>
      <td>5457</td>
    </tr>
  </tbody>
</table>
</div>



# Kesimpulan 

Dari informasi keseluruhan pada dataset, kita memiliki 10000 rows dengan 14 columns. dapat kita lihat terdapat 9% data yang hilang pada column "Tenure" dengan value yang hilang secara random / missing at random (MAR). lalu untuk menangani nilai yang hilang kita dapat drop value yang hilang atau mengisinya dengan median, serta kita akan mengubah tipe data dari float ke interger pada columns "Tenure", "Balance" dan "EstimatedSalary.


# Mempersiapkan Data

# Menangani nilai yang hilang pada column 'Tenure'

Utuk menangani nilai yang hilang pada column 'Tenure', pertama kita akan mendapatkan nilai yang unik berdasarkan column 'Surename', lalu kita akan memilih nilai acak dari list 'surename' lalu mendrop nilai yang hilang pada column tenure lalu mengisi nilai yang hilang dengan median dari columns tenure.


```python
# menangani nilai yang yang hilang pada column "Tenure"
# mendapatkan nilai unique dari nama 
#for surname in df['Surname'].unique().tolist():
    # mendaptkan nama spesifik untuk nilai 'Tenure'
   # specific_surname_df = df[df['Surname'] == surname].dropna()['Tenure']
   # surname_tenure_list = specific_surname_df.unique().tolist()
    # filter nilai yang hilang untuk mendapatkan nilai acak pada 'Tenure' berdasarkan surename. dan mengisi nilai yang hilang dengan medi
   # if surname_tenure_list != []:
       # df.loc[(df['Surname'] == surname) & (df['Tenure'] != df['Tenure']), 'Tenure'] = random.choice(surname_tenure_list)
    #else:
       # df.loc[(df['Surname'] == surname) & (df['Tenure'] != df['Tenure']), 'Tenure'] = df['Tenure'].median()
```


```python
# menangani nilai yang yang hilang pada column "Tenure" dengan mengisi nilai yang hilang dengan median
df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
```

Kita telah berhasil mengisi nilai yang hilang pada column Tenure berdasarkan nilai mediannya, mari kita lihat distribusi statistik pada dataframe.


```python
# melihat distribusi statistik setelah mengisi nilai yang hilang 
df.describe()
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
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.00000</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5000.50000</td>
      <td>1.569094e+07</td>
      <td>650.528800</td>
      <td>38.921800</td>
      <td>4.99790</td>
      <td>76485.889288</td>
      <td>1.530200</td>
      <td>0.70550</td>
      <td>0.515100</td>
      <td>100090.239881</td>
      <td>0.203700</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2886.89568</td>
      <td>7.193619e+04</td>
      <td>96.653299</td>
      <td>10.487806</td>
      <td>2.76001</td>
      <td>62397.405202</td>
      <td>0.581654</td>
      <td>0.45584</td>
      <td>0.499797</td>
      <td>57510.492818</td>
      <td>0.402769</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.556570e+07</td>
      <td>350.000000</td>
      <td>18.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>11.580000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2500.75000</td>
      <td>1.562853e+07</td>
      <td>584.000000</td>
      <td>32.000000</td>
      <td>3.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>51002.110000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5000.50000</td>
      <td>1.569074e+07</td>
      <td>652.000000</td>
      <td>37.000000</td>
      <td>5.00000</td>
      <td>97198.540000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>100193.915000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7500.25000</td>
      <td>1.575323e+07</td>
      <td>718.000000</td>
      <td>44.000000</td>
      <td>7.00000</td>
      <td>127644.240000</td>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>149388.247500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10000.00000</td>
      <td>1.581569e+07</td>
      <td>850.000000</td>
      <td>92.000000</td>
      <td>10.00000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>199992.480000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('memeriksa nilai yang hilang / na pada dataset :')
df.isna().sum()
```

    memeriksa nilai yang hilang / na pada dataset :





    RowNumber          0
    CustomerId         0
    Surname            0
    CreditScore        0
    Geography          0
    Gender             0
    Age                0
    Tenure             0
    Balance            0
    NumOfProducts      0
    HasCrCard          0
    IsActiveMember     0
    EstimatedSalary    0
    Exited             0
    dtype: int64




```python
# mengubah tipedata ke tipedata yang tepat
def convert_to_type(df, cols, type_val):
    for col in cols:
        df[col] = df[col].astype(type_val)
        
convert_to_type(df, ['Surname', 'Geography', 'Gender'], str)
convert_to_type(df, ['RowNumber', 'CustomerId', 'CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited'], 'int64')
convert_to_type(df, ['Balance', 'EstimatedSalary'], float)
```


```python
print('Informasi keseluruhan dari dataset :')
df.info()
```

    Informasi keseluruhan dari dataset :
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   RowNumber        10000 non-null  int64  
     1   CustomerId       10000 non-null  int64  
     2   Surname          10000 non-null  object 
     3   CreditScore      10000 non-null  int64  
     4   Geography        10000 non-null  object 
     5   Gender           10000 non-null  object 
     6   Age              10000 non-null  int64  
     7   Tenure           10000 non-null  int64  
     8   Balance          10000 non-null  float64
     9   NumOfProducts    10000 non-null  int64  
     10  HasCrCard        10000 non-null  int64  
     11  IsActiveMember   10000 non-null  int64  
     12  EstimatedSalary  10000 non-null  float64
     13  Exited           10000 non-null  int64  
    dtypes: float64(2), int64(9), object(3)
    memory usage: 1.1+ MB


Sekarang kita tidak mempunyai nilai yang hilang dan kita telah mengganti tipedata pada dataset. mengganti nilai yang hilang sepertinya opsi yang baik dari pada kita mengahapus columns yang terdapat nilai yang hilang. data telah di cleansing sekarang kita siap untuk mempersiapkan kelas-kelas feature, target dan melatih model.

## Memeriksa kesimbangan kelas-kelas yang ada dan melatih model tanpa mempertimbangkan ketidakseimbangan. 

# Mempersiapkan variabel feature & target

Pada tahap ini kita akan menentukan kelas-kelas feature dan menggunakan one-hot encoding untuk tipedata kategorikal. one-hot encoding berguna untuk mengubah tipedata kategorikan menjadi numerik. langkah pertama kita harus membuat variabel dummy dan mengaplikasikannya ke one-hot encoding untuk variable feature kategorikal. sebelumnya kita akan menghapus variabel yang tidak penting untuk digunakan pada variabel feature seperti CustomerId, RowNumber dan Surname. lalu kita kan melatih model tanpa mempertimbangkan ketidakseimbangan pada data.


```python
# menghapus variable yang tidak penting untuk digunakan sebagai features
df = df.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)
df.head()
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
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# one-hot encoding untuk kategorical feature
df_ohe = pd.get_dummies(df, drop_first=True)
df_ohe.head()
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
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
      <th>Gender_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# menentukan variabel untuk kelas features dan target
features = df_ohe.drop(['Exited'], axis=1)
target   = df_ohe['Exited'] 
```


```python
# membagi data kedalam traning dan testing 
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size = 0.20, random_state=12345)

# membagi data kedalam validation dan training
features_train, features_valid, target_train, target_valid,  = train_test_split(
    features_train, target_train, test_size=0.25, random_state=12345)# 0.25 * 0.80 = 0.20 for validation

# melihat panjang baris data yang telah di pisah
print('panjang baris features_train  {} ' .format(features_train.shape[0]) + 'persentase dataset sebesar 60% data')
print('panjang baris features_valid  {} ' .format(features_valid.shape[0]) + 'persentase dataset sebesar 20% data')
print('panjang baris features_test   {} ' .format(features_test.shape[0]) + 'persentase dataset sebesar 20% data')
```

    panjang baris features_train  6000 persentase dataset sebesar 60% data
    panjang baris features_valid  2000 persentase dataset sebesar 20% data
    panjang baris features_test   2000 persentase dataset sebesar 20% data



```python
# menentukan numerik features pada dataset 
numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
           'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
```


```python
# menstandarkan data numerik dengan features scaling 
scaler = StandardScaler()
scaler.fit(features_train[numeric])

# mengubah train set dan test set menggunggunakan transform()
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric]  = scaler.transform(features_test[numeric])

print('Panjang baris dari dataset features dan target')
print('-'*30)
print('Train features :',features_train.shape)
print('Train target   :',target_train.shape)
print('Valid features :',features_valid.shape)
print('Valid target   :',target_valid.shape)
print('Test features  :',features_test.shape)
print('Test target    :',target_test.shape)
display(features_train.head())
```

    Panjang baris dari dataset features dan target
    ------------------------------
    Train features : (6000, 11)
    Train target   : (6000,)
    Valid features : (2000, 11)
    Valid target   : (2000,)
    Test features  : (2000, 11)
    Test target    : (2000,)



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
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
      <th>Gender_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>492</th>
      <td>-0.134048</td>
      <td>-0.078068</td>
      <td>-0.369113</td>
      <td>0.076163</td>
      <td>0.816929</td>
      <td>-1.550255</td>
      <td>0.968496</td>
      <td>0.331571</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6655</th>
      <td>-1.010798</td>
      <td>0.494555</td>
      <td>-0.007415</td>
      <td>0.136391</td>
      <td>-0.896909</td>
      <td>0.645055</td>
      <td>0.968496</td>
      <td>-0.727858</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4287</th>
      <td>0.639554</td>
      <td>1.353490</td>
      <td>-1.454209</td>
      <td>0.358435</td>
      <td>-0.896909</td>
      <td>0.645055</td>
      <td>0.968496</td>
      <td>-0.477006</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>-0.990168</td>
      <td>2.116987</td>
      <td>-1.092511</td>
      <td>0.651725</td>
      <td>-0.896909</td>
      <td>0.645055</td>
      <td>0.968496</td>
      <td>-0.100232</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8178</th>
      <td>0.567351</td>
      <td>0.685430</td>
      <td>0.715982</td>
      <td>0.813110</td>
      <td>0.816929</td>
      <td>0.645055</td>
      <td>0.968496</td>
      <td>0.801922</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Barplot dari variabel target 
sns.countplot(target);
print('Persentase kelas pada variabel target')
display(target.value_counts())
```

    Persentase kelas pada variabel target



    0    7963
    1    2037
    Name: Exited, dtype: int64



    
![png](output_37_2.png)
    


Dapat kita lihat pada ketidakseimbangan pada kelas/variabel target

Kesimpulan : 

setelah berhasil menghapus variabel yang tidak penting pada variabel features dan menentukan variabel features, lalu kita menggunakan libarry dummy untuk mengubah variabel dengan tipedata kategorik ke numerik dengan fungsi one-hot encoding dengan hasil kita menambhkan kolum dimana variabel "geography dibagi ke dalam 2 bagian yakni : geography_german dan geography_spain dan column gender menjadi gender_male dengan nilainya dapat dengan mudah disimpulkan dari salah satu dari dua kolom lainnya (memiliki 1 di mana dua kolom lainnya memiliki nol, dan memiliki nol di tempat lain), dengan fitur ini kita tidak akan jatuh kedalam dummy trap. lalu kita telah melakukan penskalaan fitur dimana kita menghindari dimana alogaritma akan mendeteksi nilai yang lebih besar pada suatu variabel dan variabel itu dianggap lebih penting dibandingkan dengan variable yang memiliki besaran nilai yang lebih kecil. maka dari itu kita menggunakan  scaling features untuk menstandarkan data agar semua fetures dianggap penting pada saat pengeksekusian alogaritma,  sehingga rata-ratanya menjadi 0 dan variansnya menjadi 1. lalu kita membagi ukuran features train 70% dimana terdapat 7000 baris dan 11 column, features target 30% dimana terdapat 3000 baris dan 11 kolom, sekarang kita dapat melatih model dengan tidak mempertimbangkan kelas-kelad pada variabel target.


# Fungsi untuk memplot ROC Curve & precision-recall curve


```python
# fungsi untuk plot ROC curve
def plot_roc(y_test, preds, ax=None, label='model'):
    with plt.style.context('seaborn-whitegrid'):
        if not ax: fig, ax = plt.subplots(1, 1)
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        ax.plot([0, 1], [0, 1],'r--')
        ax.plot(fpr, tpr, lw=2, label=label)
        ax.legend(loc='lower right')
        ax.set_title(
            'ROC curve\n'
            f""" AP: {average_precision_score(
                y_test, preds, pos_label=1
            ):.2} | """
            f'AUC: {auc(fpr, tpr):.2}'
        )
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.annotate(f'AUC: {auc(fpr, tpr):.2}', xy=(.43, .025))
        ax.legend()
        ax.grid()
        return ax
    
# fungsi untuk plot precision-recall curve
def plot_pr(y_test, preds, ax=None, label='model'):
    with plt.style.context('seaborn-whitegrid'):
        precision, recall, thresholds = precision_recall_curve(y_test, preds)
        if not ax: fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0],'r--')    
        ax.plot(recall, precision, lw=2, label=label)
        ax.legend()
        ax.set_title(
            'Precision-recall curve\n'
            f""" Average Precision Score : {average_precision_score(
                y_test, preds, pos_label=1
            ):.2} | """
            f'AUC: {auc(recall, precision):.2}'
        )
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid()
        return ax
```

# Fungsi untuk menghitung precision, recal dan F1 Score


```python
# Fungsi untuk menghitung precision, recal dan F1 Score
def print_model_evaluation(y_test, test_predictions):
    print("\033[1m" + 'F1 score: ' + "\033[0m", '{:.3f}'.format(f1_score(y_test, test_predictions)))
    print("\033[1m" + 'Accuracy Score: ' + "\033[0m", '{:.2%}'.format(accuracy_score(y_test, test_predictions)))
    print("\033[1m" + 'Precision: ' + "\033[0m", '{:.3f}'.format(precision_score(y_test, test_predictions)))
    print("\033[1m" + 'Recall: ' + "\033[0m", '{:.3f}'.format(recall_score(y_test, test_predictions)))
    print("\033[1m" + 'Balanced Accuracy Score: ' + "\033[0m", '{:.2%}'.format(balanced_accuracy_score(y_test, test_predictions)))
    print("\033[1m" + 'Receiver Operating Characteristic(ROC) Area Under Curve (AUC) Score: ' + "\033[0m", '{:.2%}'.format(roc_auc_score(y_test, test_predictions)))
    print()
    print("\033[1m" + 'Confusion Matrix' + "\033[0m")
    print('-'*50)
    print(confusion_matrix(y_test, test_predictions))
    print()
    print("\033[1m" + 'Classification report' + "\033[0m")
    print('-'*50)
    display(pd.DataFrame(classification_report(target_test, test_predictions, output_dict=True)).T)
    print()
    
```

# Melatih model tanpa mempertimbangkan ketidakseimbangan kelasnya

# Baseline Model 


```python
# Baseline model menggunakan dummy classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(features_train, target_train)
dummy_clf_valid_predictions = dummy_clf.predict(target_valid)
plot_confusion_matrix(estimator=dummy_clf, X=features_valid, y_true=target_valid,
                      normalize='true', cmap='Blues');
```


    
![png](output_46_0.png)
    



```python
# Barplot dari variabel target 
sns.countplot(target_valid);
print('Persentase kelas pada variabel target_valid')
display(target_valid.value_counts())
```

    Persentase kelas pada variabel target_valid



    0    1609
    1     391
    Name: Exited, dtype: int64



    
![png](output_47_2.png)
    



```python
print_model_evaluation(target_valid, dummy_clf_valid_predictions)
```

    [1mF1 score: [0m 0.000
    [1mAccuracy Score: [0m 80.45%
    [1mPrecision: [0m 0.000
    [1mRecall: [0m 0.000
    [1mBalanced Accuracy Score: [0m 50.00%
    [1mReceiver Operating Characteristic(ROC) Area Under Curve (AUC) Score: [0m 50.00%
    
    [1mConfusion Matrix[0m
    --------------------------------------------------
    [[1609    0]
     [ 391    0]]
    
    [1mClassification report[0m
    --------------------------------------------------



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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.786500</td>
      <td>1.0000</td>
      <td>0.880493</td>
      <td>1573.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>427.0000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.786500</td>
      <td>0.7865</td>
      <td>0.786500</td>
      <td>0.7865</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.393250</td>
      <td>0.5000</td>
      <td>0.440246</td>
      <td>2000.0000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.618582</td>
      <td>0.7865</td>
      <td>0.692507</td>
      <td>2000.0000</td>
    </tr>
  </tbody>
</table>
</div>


    


Dengan menggunakan prediksi model dummy tanpa memperhitungkan ketidakseimbangannya, dapat kita lihat bahwa accuracy score nya tinggi yakni 80.45%% dengan skor F1 test setnya adalah 0,0 sedangkan skor F1 minimal 0,59 untukt test set. ini menunjukkan bahwa akurasi yang tinggi bukanlah ukuran yang baik untuk mengevaluasi kinerja model. 

# Logistic Regression


```python
# Membuat Fungsi untuk model logistic regression model 
def logistic_regression(X_train, y_train, X_valid, y_valid):
    model = LogisticRegression(random_state=12345, solver='liblinear')
    model.fit(X_train, y_train) # melatih model
    train_predictions = model.predict(X_train) # prediksi score dari training dataset
    valid_predictions = model.predict(X_valid) # prediksi score dari validation dataset

    # metric evaluasi dari model LogisticRegression
    print_model_evaluation(y_valid, valid_predictions)
    
```

# Sanity Check


```python
# sanity check
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train) # train the model 
test_predictions = pd.Series(model.predict(features_test))
class_frequency = test_predictions.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar');
```

    0    0.928
    1    0.072
    dtype: float64



    
![png](output_53_1.png)
    


Pada bagian ini, kita melatih model tanpa memperhitungkan ketidakseimbangan kelasnya. Kita mencapai skor F1 0,297. lalu melakukan sanity check dengan memeriksa seberapa sering kelas target berisi kelas "1" atau "0". Kita dapat mengamati ketidakseimbangan kelas pada features test yang diprediksi. Selanjutnya kita akan mencoba meningkatkan kualitas model menggunakan dua pendekatan berbeda untuk memperbaiki ketidakseimbangan kelas.

## Meningkatkan Kualitas Model 

Pada Tahap ini kita akan menggunakan dua pendekatan untuk memperbaiki ketidakseimbangan pada kelas yakni :

1. Class Weight Adjusment / Penyeimbangan Kelas
2. Upsampling

# Penyeimbangan Kelas


```python
# class weight adjustment
model = LogisticRegression(random_state=12345, class_weight='balanced', solver='liblinear')
model.fit(features_train, target_train)
train_predictions = model.predict(features_train)
valid_predictions = model.predict(features_valid)
print('F1 score dengan menggunakan Class Weight Adjustment pada train_set : {:.3f}'.format(f1_score(target_train, train_predictions)))
print('F1 score dengan menggunakan Class Weight Adjustment pada valid_set : {:.3f}'.format(f1_score(target_valid, valid_predictions)))
```

    F1 score dengan menggunakan Class Weight Adjustment pada train_set : 0.495
    F1 score dengan menggunakan Class Weight Adjustment pada valid_set : 0.475



```python
# sanity check setelah dilakukan penyeimbangan kelas
test_predictions = pd.Series(model.predict(features_test))
class_frequency = test_predictions.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar');
```

    0    0.6175
    1    0.3825
    dtype: float64



    
![png](output_58_1.png)
    


pada tahap ini kita telah menyeimbangkan kelas dengan menggunakan fungsi class_weight='balance'. dapat dilihat Score F1 nya meningkat menjadi 0.492 untuk kelas yang seimbang.

# Upsampling


```python
# fungsi untuk upsampling 
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )
    return features_upsampled, target_upsampled

# membuat training set yang baru 
features_upsampled, target_upsampled = upsample(
    features_train, target_train, 5
)
```


```python
# F1 score setelah dilakukan upsampling 
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_upsampled, target_upsampled)
train_predictions = model.predict(features_train)
valid_predictions = model.predict(features_valid)
print('F1 score setelah dilakukan upsampling: {:.3f}'.format(f1_score(target_train, train_predictions)) + ' pada train_set')
print('F1 score setelah dilakukan upsampling: {:.3f}'.format(f1_score(target_valid, valid_predictions)) + ' pada valid_set')
```

    F1 score setelah dilakukan upsampling: 0.480 pada train_set
    F1 score setelah dilakukan upsampling: 0.470 pada valid_set



```python
# sanity check setelah dilakukan upsampling
test_predictions = pd.Series(model.predict(features_test))
class_frequency = test_predictions.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar');
```

    0    0.5315
    1    0.4685
    dtype: float64



    
![png](output_63_1.png)
    


Pada tahap ini, pertama kita membagi sampel training set ke negatif dan positif, lalu kita menduplikat pengamatan positif dan menggabungkannya dengan kelas negatif. lalu kita mengacak data menggunakan fungsi shuffle()  dan melatih model dengan menggunakan LogisticRegression model dengan data yang baru. setelah itu kita menghitung score F1 setelah melakukan upsampling dengan score 0.484, terdapat peningkatan dari sebelumnya dimana skor F1 sebelum mempertimbangkan keseimbangan kelas yakni 0.300. 

# Memilih Parameter Terbaik



Pada tahap ini dengan meggunakan training set, kita akan memilih parameter terbaik dari beberapa model untuk dibandingkan, model mana yang paling terbaik untuk prediksi kita.

# Decision Tree Classifier


```python
# optimasisasi hyperparameter untuk model Decision tree classifier
parameters = {
    "criterion" : ["gini", "entropy"],
    "max_depth" : [2, 4, 8, 16],
    "min_samples_split" : [2, 4, 8, 16],
    "min_samples_leaf" : [2, 4, 6]
    }
classifier = DecisionTreeClassifier()
grid = GridSearchCV(classifier, parameters, scoring='f1', cv=5)
grid.fit(features_train, target_train) 
y_pred = grid.predict(features_valid)
print('Kombinasi parameter yang memberikan kita skor F1 : ')
print(grid.best_params_)
print('Accuracy terbaik setelah menentukan parameter terbaik : {:.3f}'.format(grid.best_score_))
print('Accuracy dari model terbaik pada training dataset : {:.3f}'.format(grid.score(features_train, target_train)))
print('F1 score: ', f1_score(target_valid, y_pred))
```

    Kombinasi parameter yang memberikan kita skor F1 : 
    {'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 16}
    Accuracy terbaik setelah menentukan parameter terbaik : 0.576
    Accuracy dari model terbaik pada training dataset : 0.664
    F1 score:  0.5435114503816795



```python
# membuat fungsi decision tree classifier
def decision_tree_classifier(X_train, y_train, X_valid, y_valid):
    
    # membuat list untuk mendapatkan nilai
    train_scores = []
    valid_scores = []
    f1_scores = []
    
    # Melatih Model
    model = DecisionTreeClassifier(**grid.best_params_) 
    model.fit(X_train, y_train) # melatih model
   
    # membuat prediksi dari train set
    train_predictions = model.predict(X_train)
    train_predictions_acc = accuracy_score(y_train, train_predictions)
    train_scores.append(train_predictions_acc)
    
    # membuat prediksi dari valid set
    dt_valid_predictions = model.predict(X_valid) 
    dt_valid_predictions_acc = accuracy_score(y_valid, dt_valid_predictions)
    valid_scores.append(dt_valid_predictions_acc)
    f1_score_ = f1_score(y_valid, dt_valid_predictions)
    f1_scores.append(f1_score_)
    scores = list(zip(f1_scores, train_scores, valid_scores))
    print('Pengklasifikasian terbaik dari model  Decission tree classification dengan ' "\033[1m" + 'F1 score {:.3f}'.format(max(scores, key = lambda x: x[0])[0]) + "\033[0m" +   
          ' dan score accuracy ' "\033[1m" '{:.2%}'.format(max(scores, key = lambda x: x[0])[1]) + ' untuk training set' + "\033[0m" + 
          ' dan ' + "\033[1m" '{:.2%}'.format(max(scores, key = lambda x: x[0])[2]) + ' untuk valid set' + "\033[0m")
    print()
    
    # metric evaluasi dari model DecisionTreeClassifier
    print_model_evaluation(target_valid, dt_valid_predictions)
    

```


```python
# Menjalankan fungsi dari model decision tree classifier
decision_tree_classifier(features_train, target_train, features_valid, target_valid)
```

    Pengklasifikasian terbaik dari model  Decission tree classification dengan [1mF1 score 0.544[0m dan score accuracy [1m88.53% untuk training set[0m dan [1m85.05% untuk valid set[0m
    
    [1mF1 score: [0m 0.544
    [1mAccuracy Score: [0m 85.05%
    [1mPrecision: [0m 0.674
    [1mRecall: [0m 0.455
    [1mBalanced Accuracy Score: [0m 70.09%
    [1mReceiver Operating Characteristic(ROC) Area Under Curve (AUC) Score: [0m 70.09%
    
    [1mConfusion Matrix[0m
    --------------------------------------------------
    [[1523   86]
     [ 213  178]]
    
    [1mClassification report[0m
    --------------------------------------------------



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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.783410</td>
      <td>0.864590</td>
      <td>0.822001</td>
      <td>1573.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.193182</td>
      <td>0.119438</td>
      <td>0.147612</td>
      <td>427.0000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.705500</td>
      <td>0.705500</td>
      <td>0.705500</td>
      <td>0.7055</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.488296</td>
      <td>0.492014</td>
      <td>0.484806</td>
      <td>2000.0000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.657396</td>
      <td>0.705500</td>
      <td>0.678019</td>
      <td>2000.0000</td>
    </tr>
  </tbody>
</table>
</div>


    


Pada tahap ini dengan menggunakan library GridSearchCV untuk melakukan optimasi pada hyperparameter pada paramter max_depth, criteria, min_sample_split, dan min_sample_leaf. kita menentukan parameter terbaik dari model decision tree classifier. dimana pada kedalaman/depth yang dangkal dari model pada umumnya tidak overfit tetapi memiliki kinerja yang buruk (bias yang tinggi dan varian yang rendah), dan depth yang dalam pada umumnya overfit dan memiliki kinerja yang baik (bias yang rendah dan varian yang tinggi), sehingga kedalam/depth yang kita inginkan adalah yang tidak terlalu dangkan sehingga memiliki kinerja rendah dan tidak terlalu dalam sehingga membuat overfit pada training dataset. kita perlu memiliki keseimbangan antara bias dan variannya. dimana pada kedalaman/ max_depth = 8 kita mendapatkan F1 score 0.548, dengan score accuracy 87.94% untuk  training set dan 84.57% untuk testing set.

# Logistic Regression Model


```python
# optimasi parameter model Logistic Regression 
# menentukan parameters
grid = {
    "solver" : ['newton-cg', 'lbfgs', 'liblinear'],
    "penalty" : ['l2'],
    "C" : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# menentukan grid search
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# melatih model
regressor = LogisticRegression()
grid_search = GridSearchCV(estimator = regressor, param_grid = grid,  n_jobs=-1, cv=cv, scoring='f1', error_score=0)
grid_search.fit(features_train, target_train) 
y_pred = grid_search.predict(features_valid)
print('Kombinasi parameter yang memberikan kita skor F1 terbaik : ')
print(grid_search.best_params_)
print('Accuracy terbaik setelah menentukan parameter terbaik via grid search : {:.3f}'.format(grid_search.best_score_))
print('Accuracy dari model terbaik pada training dataset : {:.3f}'.format(grid_search.score(features_train, target_train)))
print('F1 score: ', f1_score(target_valid, y_pred))
```

    Kombinasi parameter yang memberikan kita skor F1 terbaik : 
    {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
    Accuracy terbaik setelah menentukan parameter terbaik via grid search : 0.323
    Accuracy dari model terbaik pada training dataset : 0.328
    F1 score:  0.30451127819548873



```python
# membuat fungsi logistic regression 
def logistic_regression(X_train, y_train, X_valid, y_valid):
    
    # membuat list untuk mendapatkan nilai
    train_scores = []
    valid_scores = []
    f1_scores = []
    
    # Melatih Model
    model = LogisticRegression(**grid_search.best_params_)
    model.fit(X_train, y_train) # melatih model
   
    # membuat prediksi dari train set
    train_predictions = model.predict(X_train)
    train_predictions_acc = accuracy_score(y_train, train_predictions)
    train_scores.append(train_predictions_acc)
    
    # membuat prediksi dari valid set
    lr_valid_predictions = model.predict(X_valid) 
    lr_valid_predictions_acc = accuracy_score(y_valid, lr_valid_predictions)
    valid_scores.append(lr_valid_predictions_acc)
    f1_score_ = f1_score(y_valid, lr_valid_predictions)
    f1_scores.append(f1_score_)
    scores = list(zip(f1_scores, train_scores, valid_scores))
    print('Parameter yang memberikan kita skor terbaik untuk ' "\033[1m" + 
          'F1 score of {:.3f} '.format(max(scores, key = lambda x: x[0])[0]) + "\033[0m" + 
          'menggunakan' "\033[1m" + ' C parameter  {},'.format(grid_search.best_params_['C']) + "\033[0m" +
          "\033[1m" + ' {} sebagai logistic regression solver'.format(grid_search.best_params_['solver']) + "\033[0m" +
          ' mengarah ke nilai accuracy ' "\033[1m" + '{:.2%}'.format(max(scores, key = lambda x: x[0])[1]) + ' untuk training set ' + "\033[0m" + 
          'dan ' + "\033[1m" '{:.2%}'.format(max(scores, key = lambda x: x[0])[2]) + ' untuk valid set' + "\033[0m")
    print()
    
    # metric evaluasi dari model logistic_regression
    print_model_evaluation(y_valid, lr_valid_predictions)
    

```


```python
logistic_regression(features_train, target_train, features_valid, target_valid)
```

    Parameter yang memberikan kita skor terbaik untuk [1mF1 score of 0.305 [0mmenggunakan[1m C parameter  10,[0m[1m newton-cg sebagai logistic regression solver[0m mengarah ke nilai accuracy [1m81.28% untuk training set [0mdan [1m81.50% untuk valid set[0m
    
    [1mF1 score: [0m 0.305
    [1mAccuracy Score: [0m 81.50%
    [1mPrecision: [0m 0.574
    [1mRecall: [0m 0.207
    [1mBalanced Accuracy Score: [0m 58.49%
    [1mReceiver Operating Characteristic(ROC) Area Under Curve (AUC) Score: [0m 58.49%
    
    [1mConfusion Matrix[0m
    --------------------------------------------------
    [[1549   60]
     [ 310   81]]
    
    [1mClassification report[0m
    --------------------------------------------------



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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.785368</td>
      <td>0.928163</td>
      <td>0.850816</td>
      <td>1573.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.198582</td>
      <td>0.065574</td>
      <td>0.098592</td>
      <td>427.000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.744000</td>
      <td>0.744000</td>
      <td>0.744000</td>
      <td>0.744</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.491975</td>
      <td>0.496868</td>
      <td>0.474704</td>
      <td>2000.000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.660089</td>
      <td>0.744000</td>
      <td>0.690216</td>
      <td>2000.000</td>
    </tr>
  </tbody>
</table>
</div>


    


Pada tahap ini kita menyetel parameter "C" untuk model regresi logistik. Meskipun pelatihan modelnya cepat, skor F1 lebih rendah yaitu 0,306. Model regresi logistik memberikan akurasi 81.50% untuk training set, dan  80.23% untuk testing set saat menggunakan parameter "C" 10. Kita dapat melihat di sini bahwa score pada training set maupun test set tidak cukup tinggi. Ini karena modelnya tidak cukup kompleks sehingga terjadi underfitting. Mari kita lihat bagaimana hasil prediksi model lain sebelum memutuskan model terbaik yang akan digunakan untuk memprediksi.

# Catboost Classifier


```python
# membuat fungsi catboost classifier
def catboost_classifier(X_train, y_train, X_valid, y_valid):
    
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
    
    # Melatih Model
    model = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight ,random_state=12345)
    model.fit(X_train, y_train)
   
    # membuat prediksi dari train set
    train_predictions = model.predict(X_train)
    train_predictions_acc = accuracy_score(y_train, train_predictions)
    
    # membuat prediksi dari testing set
    cb_valid_predictions = model.predict(X_valid) 
    cb_valid_predictions_acc = accuracy_score(y_valid, cb_valid_predictions)
    f1_score_ = f1_score(y_valid, cb_valid_predictions)
    print('Model Catboost Classifier memiliki ' "\033[1m" 'F1 score {:.3f},'.format(f1_score_) + "\033[0m" +
          ' Accuracy Scor ' "\033[1m" + '{:.2%}'.format(train_predictions_acc) + ' untuk training set ' + "\033[0m" + 
          'dan ' + "\033[1m" '{:.2%}'.format(cb_valid_predictions_acc) + ' untuk valid set' + "\033[0m")
    print()
   
    # metric evaluasi dari model catboost classifier
    print_model_evaluation(y_valid, cb_valid_predictions)
    

```


```python
catboost_classifier(features_train, target_train, features_valid, target_valid)
```

    Model Catboost Classifier memiliki [1mF1 score 0.600,[0m Accuracy Scor [1m89.53% untuk training set [0mdan [1m81.50% untuk valid set[0m
    
    [1mF1 score: [0m 0.600
    [1mAccuracy Score: [0m 81.50%
    [1mPrecision: [0m 0.520
    [1mRecall: [0m 0.711
    [1mBalanced Accuracy Score: [0m 77.56%
    [1mReceiver Operating Characteristic(ROC) Area Under Curve (AUC) Score: [0m 77.56%
    
    [1mConfusion Matrix[0m
    --------------------------------------------------
    [[1352  257]
     [ 113  278]]
    
    [1mClassification report[0m
    --------------------------------------------------



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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.791126</td>
      <td>0.736809</td>
      <td>0.763002</td>
      <td>1573.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.226168</td>
      <td>0.283372</td>
      <td>0.251559</td>
      <td>427.00</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.640000</td>
      <td>0.640000</td>
      <td>0.640000</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.508647</td>
      <td>0.510091</td>
      <td>0.507281</td>
      <td>2000.00</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.670508</td>
      <td>0.640000</td>
      <td>0.653809</td>
      <td>2000.00</td>
    </tr>
  </tbody>
</table>
</div>


    


Kesimpulan : 
    
Dari investigasi kualitas model yang berbeda, kita dapat melihat bahwa pengklasifikasi CatBoost memberikan hasil terbaik untuk skor F1 0.600, akurasi 81.83% dan nilai AUC-ROC 77.82%  dari empat model berbeda yang diselidiki. Model regresi logistik memiliki nilai terendah untuk skor F1 sebesar 0.304, Accuracy Scor 89.53% untuk training set dan 81.50% untuk valid set. Model Catboost adalah model terbaik berdasarkan skor F1 saat kita melatih model dengan hyperparameter terbak saat menggunakan train set dan valid set.

## menjalankan pengujian terakhir

Pengujian model :

Hasil dari bagian sebelumnya menunjukkan bahwa pengklasifikasi CatBoost mungkin merupakan model yang paling akurat. Menggunakan pengklasifikasi CatBoost sebagai model akhir kita, kita dapat membuat prediksi menggunakan set pengujian.


```python
# membuat fungsi catboost classifier
def catboost_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test):
    
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
    
    # Melatih Model
    model = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight ,random_state=12345)
    model.fit(X_train, y_train)
   
    # membuat prediksi dari train set
    train_predictions = model.predict(X_train)
    train_predictions_acc = accuracy_score(y_train, train_predictions)
    
    # membuat prediksi dari train set
    valid_predictions = model.predict(X_valid)
    valid_predictions_acc = accuracy_score(y_valid, valid_predictions)
    
    # membuat prediksi dari testing set
    cb_test_predictions = model.predict(X_test) 
    cb_test_predictions_acc = accuracy_score(y_test, cb_test_predictions)
    f1_score_ = f1_score(y_test, cb_test_predictions)
   
    # metric evaluasi dari model catboost classifier
    print_model_evaluation(target_test, cb_test_predictions)
    
     # plot dari Curve ROC dan Precision-Recall curv
    _, axs = plt.subplots(1, 2,figsize=(10,5))
    axs = axs.ravel()
    plot_pr(y_test, cb_test_predictions, ax=axs[0], label="CatBoostClassifier")
    plot_roc(y_test, cb_test_predictions, ax=axs[1], label="CatBoostClassifier")
    

```


```python
catboost_classifier(features_train, target_train, features_valid, target_valid, features_test, target_test)
```

    [1mF1 score: [0m 0.638
    [1mAccuracy Score: [0m 82.45%
    [1mPrecision: [0m 0.570
    [1mRecall: [0m 0.724
    [1mBalanced Accuracy Score: [0m 78.78%
    [1mReceiver Operating Characteristic(ROC) Area Under Curve (AUC) Score: [0m 78.78%
    
    [1mConfusion Matrix[0m
    --------------------------------------------------
    [[1340  233]
     [ 118  309]]
    
    [1mClassification report[0m
    --------------------------------------------------



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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.919067</td>
      <td>0.851875</td>
      <td>0.884197</td>
      <td>1573.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.570111</td>
      <td>0.723653</td>
      <td>0.637771</td>
      <td>427.0000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.824500</td>
      <td>0.824500</td>
      <td>0.824500</td>
      <td>0.8245</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.744589</td>
      <td>0.787764</td>
      <td>0.760984</td>
      <td>2000.0000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.844565</td>
      <td>0.824500</td>
      <td>0.831585</td>
      <td>2000.0000</td>
    </tr>
  </tbody>
</table>
</div>


    



    
![png](output_84_3.png)
    


Dengan Menggunakan Model Catboost model, kita mendapatkan F1-score 0.638

## Kesimpulan Akhir 


1. Dari informasi keseluruhan pada dataset, kita memiliki 10000 rows dengan 14 columns. dapat kita lihat terdapat 9% data yang hilang pada column "Tenure" dengan value yang hilang secara random / missing at random (MAR). lalu untuk menangani nilai yang hilang kita dapat drop value yang hilang atau mengisinya dengan median, serta kita akan mengubah tipe data dari float ke interger pada columns "Tenure", "Balance" dan "EstimatedSalary.
-----------
2. Utuk menangani nilai yang hilang pada column 'Tenure', pertama kita akan mendapatkan nilai yang unik berdasarkan column 'Surename', lalu kita akan memilih nilai acak dari list 'surename' lalu mendrop nilai yang hilang pada column tenure lalu mengisi nilai yang hilang dengan median dari columns tenure.
---------
3. Mempersiapkan variabel feature & target
Pada tahap ini kita akan menentukan kelas-kelas feature dan menggunakan one-hot encoding untuk tipedata kategorikal. one-hot encoding berguna untuk mengubah tipedata kategorikan menjadi numerik. langkah pertama kita harus membuat variabel dummy dan mengaplikasikannya ke one-hot encoding untuk variable feature kategorikal. sebelumnya kita akan menghapus variabel yang tidak penting untuk digunakan pada variabel feature seperti CustomerId, RowNumber dan Surname. lalu kita kan melatih model tanpa mempertimbangkan ketidakseimbangan pada data.
-----------
4.Meningkatkan Kualitas Model
Pada Tahap ini kita akan menggunakan dua pendekatan untuk memperbaiki ketidakseimbangan pada kelas yakni :

a. Class Weight Adjusment / Penyeimbangan Kelas
b. Upsampling

c. Memilih Parameter Terbaik
Pada tahap ini dengan meggunakan training set, kita akan memilih parameter terbaik dari beberapa model untuk dibandingkan, model mana yang paling terbaik untuk prediksi kita. lalu Dari investigasi kualitas model yang berbeda, kita dapat melihat bahwa pengklasifikasi CatBoost memberikan hasil terbaik untuk skor F1 0,60, akurasi 81.50% dan nilai AUC-ROC 77.56% dari empat model berbeda yang diselidiki. Model regresi logistik memiliki nilai terendah untuk skor F1 sebesar 0.304, akurasi sebesar 80.13%, dan AUC-ROC sebesar 58.28%. Model Catboost adalah model terbaik berdasarkan skor F1 saat memprediksi apakah nasabah akan meninggalkan bank.

