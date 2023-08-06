# Deskripsi Proyek 

Perusahaan Asuransi bernama "Sure Tomorrow" ingin menyelesaikan beberapa masalah dengan bantuan machine learning. Kamu diminta untuk mengevaluasi kemungkinan itu.

- Tugas 1: Temukan klien yang mirip dengan kriteria klien tertentu. Tugas ini akan memudahkan perusahaan untuk melakukan pemasaran.
- Tugas 2: Prediksi apakah klien baru cenderung mendapatkan manfaat asuransi. Apakah prediksi model lebih baik daripada prediksi model *dummy*?
- Tugas 3: Prediksi jumlah manfaat asuransi yang mungkin diterima klien baru dengan menggunakan model regresi linear.
- Tugas 4: Lindungi data pribadi klien tanpa merusak model dari tugas sebelumnya. Penting untuk mengembangkan algoritma transformasi data yang dapat mencegah penyalahgunaan informasi pribadi jika data tersebut jatuh ke pihak yang salah. Hal ini disebut penyembunyian data atau pengaburan data. Namun, prosedur perlindungan datanya pun perlu diperhatikan agar kualitas model machine learning tidak menurun. Kamu tidak perlu memilih model terbaik, cukup buktikan bahwa algoritma bekerja secara akurat.

# Pra-pemrosesan & Eksplorasi Data

## Inisialisasi


```python
pip install scikit-learn --upgrade
```

    [33mWARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
    [33mWARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
    Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.9/site-packages (0.24.1)
    Collecting scikit-learn
      Using cached scikit_learn-1.2.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.6 MB)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.9/site-packages (from scikit-learn) (3.1.0)
    Collecting joblib>=1.1.1
      Using cached joblib-1.2.0-py3-none-any.whl (297 kB)
    Requirement already satisfied: scipy>=1.3.2 in /opt/conda/lib/python3.9/site-packages (from scikit-learn) (1.8.0)
    Requirement already satisfied: numpy>=1.17.3 in /opt/conda/lib/python3.9/site-packages (from scikit-learn) (1.21.1)
    [33mWARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
    Installing collected packages: joblib, scikit-learn
      Attempting uninstall: joblib
    [33m    WARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
        Found existing installation: joblib 1.1.0
        Uninstalling joblib-1.1.0:
    [31mERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: 'INSTALLER'
    Consider using the `--user` option or check the permissions.
    [0m
    [33mWARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
    [33mWARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
    [33mWARNING: Ignoring invalid distribution -oblib (/opt/conda/lib/python3.9/site-packages)[0m
    Note: you may need to restart the kernel to use updated packages.



```python
import numpy as np
import pandas as pd
import math

import seaborn as sns

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

from IPython.display import display
```

## Muat Data

Muat data dan lakukan pemeriksaan untuk memastikan data bebas dari permasalahan.


```python
df = pd.read_csv('/datasets/insurance_us.csv')
```

Ganti nama kolom agar kode terlihat lebih konsisten.


```python
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
```


```python
df.sample(10)
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>insurance_benefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1367</th>
      <td>1</td>
      <td>45.0</td>
      <td>41200.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>1</td>
      <td>44.0</td>
      <td>57300.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4391</th>
      <td>0</td>
      <td>30.0</td>
      <td>18200.0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>1</td>
      <td>30.0</td>
      <td>40500.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3586</th>
      <td>0</td>
      <td>30.0</td>
      <td>33700.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>18.0</td>
      <td>46900.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3798</th>
      <td>1</td>
      <td>23.0</td>
      <td>50400.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4583</th>
      <td>0</td>
      <td>30.0</td>
      <td>44100.0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3350</th>
      <td>1</td>
      <td>22.0</td>
      <td>38800.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2537</th>
      <td>0</td>
      <td>33.0</td>
      <td>66100.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 5 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   gender              5000 non-null   int64  
     1   age                 5000 non-null   float64
     2   income              5000 non-null   float64
     3   family_members      5000 non-null   int64  
     4   insurance_benefits  5000 non-null   int64  
    dtypes: float64(2), int64(3)
    memory usage: 195.4 KB



```python
# Mungkin kamu ingin mengganti tipe usia (dari float ke int) meskipun tidak terlalu penting untuk dilakukan

# ketik konversi di sini jika kamu memilih:
df['age'] = df['age'].astype(int)
```


```python
# periksa apakah konversinya sudah berhasil
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 5 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   gender              5000 non-null   int64  
     1   age                 5000 non-null   int64  
     2   income              5000 non-null   float64
     3   family_members      5000 non-null   int64  
     4   insurance_benefits  5000 non-null   int64  
    dtypes: float64(1), int64(4)
    memory usage: 195.4 KB



```python
# amati statistik deskriptif data. 
# Apakah sudah dengan benar?
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>insurance_benefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.499000</td>
      <td>30.952800</td>
      <td>39916.360000</td>
      <td>1.194200</td>
      <td>0.148000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500049</td>
      <td>8.440807</td>
      <td>9900.083569</td>
      <td>1.091387</td>
      <td>0.463183</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>5300.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>33300.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>40200.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>37.000000</td>
      <td>46600.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>65.000000</td>
      <td>79000.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>insurance_benefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>41</td>
      <td>49600.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>46</td>
      <td>38000.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>29</td>
      <td>21000.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>21</td>
      <td>41700.0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>28</td>
      <td>26100.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    gender                0
    age                   0
    income                0
    family_members        0
    insurance_benefits    0
    dtype: int64



Kesimpulan :

Persiapan preprosessing data sudah berhasil, dengan informasi dataset 5000 baris dan 5 kolom, untuk kemudahan kita juga mengubah tipe data pada kolom age dari float ke tipe interger. kita juga memeriksa tida ada nilai yang hilang pada data. sekarang kita dapat mengeksplor data menggunakan exploratory data analysis.


## EDA

Apakah ada kelompok klien tertentu dengan mengamati sepasang plot.


```python
g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)
```


    
![png](output_20_0.png)
    


Mungkin itu agak sulit untuk mendeteksi kelompok (klaster) dengan jelas karena sulit untuk menggabungkan beberapa variabel secara bersamaan (untuk menganalisis distribusi multivariat). Namun, itulah gunanya LA dan ML di sini.

# Tugas 1. Klien yang mirip

Dalam bahasa pemrograman ML, penting untuk mengembangkan prosedur yang bisa menentukan k *neighbors* (objek) terdekat pada objek tertentu berdasarkan jarak pada objeknya.

Kamu mungkin ingin mengulas kembali pelajaran berikut (bab -> pelajaran)
- Jarak Antar Vektor -> Jarak Euclidean
- Jarak Antar Vektor -> Jarak Manhattan

Untuk menyelesaikannya, cobalah beberapa metrik.

Ketik fungsi yang menampilkan k *neighbors* terdekat untuk objek ke-$n^{th}$ berdasarkan metrik jarak tertentu. Di soal ini, jumlah manfaat asuransi yang diterima tidak diperhitungkan.

Gunakan implementasi algoritma kNN yang tersedia di scikit-learn (periksa [link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) atau gunakan milikmu sendiri.

Uji untuk empat kombinasi dari dua kasus
- Penskalaan
  - datanya tidak ada skalanya
  - mengatur skala data dengan [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) 
- Metrik Jarak
  - Euclidean
  - Manhattan

Jawablah pertanyaan-pertanyaan berikut:
- Apakah data yang tidak berskala memengaruhi algoritma kNN? Jika berpengaruh, lalu bagaimanakah bentuknya?
- Seberapa mirip hasil yang didapatkan dari metrik Euclidean saat menggunakan metrik jarak Manhattan (terlepas dari penskalaannya)?


```python
feature_names = ['gender', 'age', 'income', 'family_members']
```


```python
def get_knn(df, n, k, metric):
    
    """
    Tampilkan k neighbors terdekat

    :param df: pandas DataFrame digunakan untuk menemukan objek-objek yang mirip
    :param n: nomor objek yang dicari *neighbors* terdekat
    :param k: jumlah *neighbors* terdekat yang ditampilkan
    :param metric: nama metrik jarak
    """

    # < kode program di sini >
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=metric) 
    nbrs.fit(df[feature_names].values)
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res
```

Mengatur skala data.


```python
feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())
```


```python
df_scaled.sample(5)
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>insurance_benefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>471</th>
      <td>1.0</td>
      <td>0.461538</td>
      <td>0.356962</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.0</td>
      <td>0.292308</td>
      <td>0.465823</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2517</th>
      <td>0.0</td>
      <td>0.661538</td>
      <td>0.474684</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4558</th>
      <td>0.0</td>
      <td>0.569231</td>
      <td>0.420253</td>
      <td>0.166667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1852</th>
      <td>0.0</td>
      <td>0.415385</td>
      <td>0.727848</td>
      <td>0.333333</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Dapatkan catatan yang mirip untuk tiap kombinasi


```python
# jarak euclidean dengan data tidak berskala
get_knn(df[feature_names], 0, 10, 'euclidean')
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>41</td>
      <td>49600.0</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>1</td>
      <td>41</td>
      <td>49600.0</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>0</td>
      <td>42</td>
      <td>49600.0</td>
      <td>0</td>
      <td>1.732051</td>
    </tr>
    <tr>
      <th>4031</th>
      <td>1</td>
      <td>44</td>
      <td>49600.0</td>
      <td>2</td>
      <td>3.162278</td>
    </tr>
    <tr>
      <th>3424</th>
      <td>0</td>
      <td>38</td>
      <td>49600.0</td>
      <td>0</td>
      <td>3.316625</td>
    </tr>
    <tr>
      <th>815</th>
      <td>1</td>
      <td>37</td>
      <td>49600.0</td>
      <td>2</td>
      <td>4.123106</td>
    </tr>
    <tr>
      <th>4661</th>
      <td>0</td>
      <td>45</td>
      <td>49600.0</td>
      <td>0</td>
      <td>4.242641</td>
    </tr>
    <tr>
      <th>2125</th>
      <td>0</td>
      <td>37</td>
      <td>49600.0</td>
      <td>2</td>
      <td>4.242641</td>
    </tr>
    <tr>
      <th>2349</th>
      <td>1</td>
      <td>46</td>
      <td>49600.0</td>
      <td>2</td>
      <td>5.099020</td>
    </tr>
    <tr>
      <th>3900</th>
      <td>1</td>
      <td>36</td>
      <td>49600.0</td>
      <td>0</td>
      <td>5.099020</td>
    </tr>
  </tbody>
</table>
</div>




```python
# jarak euclidean dengan data berskala
get_knn(df_scaled[feature_names], 0, 10, 'euclidean')
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.627848</td>
      <td>0.166667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2689</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.634177</td>
      <td>0.166667</td>
      <td>0.006329</td>
    </tr>
    <tr>
      <th>133</th>
      <td>1.0</td>
      <td>0.615385</td>
      <td>0.636709</td>
      <td>0.166667</td>
      <td>0.017754</td>
    </tr>
    <tr>
      <th>4869</th>
      <td>1.0</td>
      <td>0.646154</td>
      <td>0.637975</td>
      <td>0.166667</td>
      <td>0.018418</td>
    </tr>
    <tr>
      <th>3275</th>
      <td>1.0</td>
      <td>0.646154</td>
      <td>0.651899</td>
      <td>0.166667</td>
      <td>0.028550</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>1.0</td>
      <td>0.615385</td>
      <td>0.602532</td>
      <td>0.166667</td>
      <td>0.029624</td>
    </tr>
    <tr>
      <th>2103</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.596203</td>
      <td>0.166667</td>
      <td>0.031646</td>
    </tr>
    <tr>
      <th>3365</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.596203</td>
      <td>0.166667</td>
      <td>0.031646</td>
    </tr>
    <tr>
      <th>124</th>
      <td>1.0</td>
      <td>0.661538</td>
      <td>0.635443</td>
      <td>0.166667</td>
      <td>0.031693</td>
    </tr>
    <tr>
      <th>3636</th>
      <td>1.0</td>
      <td>0.615385</td>
      <td>0.600000</td>
      <td>0.166667</td>
      <td>0.031815</td>
    </tr>
  </tbody>
</table>
</div>




```python
# jarak manhattan dengan data tidak berskala
get_knn(df[feature_names], 0, 10, 'manhattan')
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>41</td>
      <td>49600.0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>1</td>
      <td>41</td>
      <td>49600.0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>0</td>
      <td>42</td>
      <td>49600.0</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4031</th>
      <td>1</td>
      <td>44</td>
      <td>49600.0</td>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>815</th>
      <td>1</td>
      <td>37</td>
      <td>49600.0</td>
      <td>2</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3424</th>
      <td>0</td>
      <td>38</td>
      <td>49600.0</td>
      <td>0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2125</th>
      <td>0</td>
      <td>37</td>
      <td>49600.0</td>
      <td>2</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3900</th>
      <td>1</td>
      <td>36</td>
      <td>49600.0</td>
      <td>0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2349</th>
      <td>1</td>
      <td>46</td>
      <td>49600.0</td>
      <td>2</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4661</th>
      <td>0</td>
      <td>45</td>
      <td>49600.0</td>
      <td>0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# jarak manhattan dengan data berskala
get_knn(df_scaled[feature_names], 0, 10, 'manhattan')
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.627848</td>
      <td>0.166667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2689</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.634177</td>
      <td>0.166667</td>
      <td>0.006329</td>
    </tr>
    <tr>
      <th>133</th>
      <td>1.0</td>
      <td>0.615385</td>
      <td>0.636709</td>
      <td>0.166667</td>
      <td>0.024245</td>
    </tr>
    <tr>
      <th>4869</th>
      <td>1.0</td>
      <td>0.646154</td>
      <td>0.637975</td>
      <td>0.166667</td>
      <td>0.025511</td>
    </tr>
    <tr>
      <th>3365</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.596203</td>
      <td>0.166667</td>
      <td>0.031646</td>
    </tr>
    <tr>
      <th>2103</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.596203</td>
      <td>0.166667</td>
      <td>0.031646</td>
    </tr>
    <tr>
      <th>124</th>
      <td>1.0</td>
      <td>0.661538</td>
      <td>0.635443</td>
      <td>0.166667</td>
      <td>0.038364</td>
    </tr>
    <tr>
      <th>4305</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.588608</td>
      <td>0.166667</td>
      <td>0.039241</td>
    </tr>
    <tr>
      <th>3275</th>
      <td>1.0</td>
      <td>0.646154</td>
      <td>0.651899</td>
      <td>0.166667</td>
      <td>0.039435</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>1.0</td>
      <td>0.615385</td>
      <td>0.602532</td>
      <td>0.166667</td>
      <td>0.040701</td>
    </tr>
  </tbody>
</table>
</div>



Jawab pertanyaan

**Apakah data yang tidak berskala memengaruhi algoritma kNN? Jika berpengaruh, lalu bagaimanakah bentuknya?** 

Penskalaan fitur penting untuk algoritme kNN karena menghitung jarak antar data. Data yang tidak diskalakan memepengaruhi algoritma kNN karena algoritma lebih mementingkan fitur tertentu dengan rentang angka yang lebih tinggi daripada yang lain. Dalam latihan yang menggunakan jarak euclidean misalnya, metrik jarak yang dihitung untuk data yang tidak diskalakan berkisar antara 0,00 hingga 5,09, sedangkan untuk data yang diskalakan, kami memperoleh rentang metrik jarak dari 0,00 hingga 0,03. Ini menunjukkan dampak penskalaan pada data. kNN dengan jarak euclidean peka terhadap besaran, oleh karena itu, data harus diskalakan agar semua fitur memiliki bobot yang sama.

**Seberapa mirip hasil yang didapatkan dari metrik Euclidean saat menggunakan metrik jarak Manhattan (terlepas dari penskalaannya)?** 


Jarak terdekat yang dihasilkan pada metrik euclidian dan manhattan metrik terlepas dari penskalaannya hampir serupa pada index ke dua yakni dengan nilai 1.0 pada metrik tidak berskala dan 0.006329 pada data berskala. setelah itu jarak yang diihasilkan berbeda.

# Tugas 2. Apakah klien kemungkinan menerima manfaat asuransi?

Dalam konteks machine learning, tugas ini sama seperti tugas klasifikasi biner.

Dengan target `insurance_benefits` yang lebih dari nol, coba evaluasi apakah model klasifikasi kNN merupakan pendekatan yang lebih baik daripada model *dummy*.

Instruksi:
- Buat pengklasifikasi berbasis KNN dan ukur kualitasnya dengan metrik F1 untuk k=1..10 untuk data asli dan data yang diskalakan. Menarik untuk lihat bagaimana k dapat memengaruhi metrik evaluasi dan apakah penskalaan data membuat hasilnya berbeda. Gunakan implementasi algoritma klasifikasi kNN yang tersedia di scikit-learn (periksa [link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) atau milikmu sendiri.
- Buat model *dummy* yang acak untuk kasus ini. Model tersebut harusnya menampilkan "1" dengan beberapa probabilitas. Uji model dengan empat nilai probabilitas: 0, probabilitas membayar manfaat asuransi apa pun; 0,5; 1.

Probabilitas membayar manfaat asuransi dapat didefinisikan sebagai

$$
P\{\text{manfaat asuransi yang diterima}\}=\frac{\text{jumlah klien yang menerima manfaat asuransi}}{\text{jumlah klien secara keseluruhan}}.
$$

Pisahkan keseluruhan data menjadi 70:30 untuk proporsi **training** dan **test set**.


```python
# Hitung tagetnya

df['insurance_benefits_received'] = (df['insurance_benefits'] > 0).astype(int)# <kode program di sini>
```


```python
# periksa ketidakseimbangan kelas dengan value_counts()

# < kode program di sini >
df['insurance_benefits_received'].value_counts()
```




    0    4436
    1     564
    Name: insurance_benefits_received, dtype: int64




```python
def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# jika kamu memiliki masalah dengan baris berikut, muat ulang kernel dan jalankan notebook kembali
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Confusion Matrix')
    print(cm)
```


```python
# mengumpulkan output pada model acak

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)
```

Mengukur metrik F1 pada data asli menggunakan klasifikasi KNN



```python
# Memisahkan dataset ke train dan test set
train, test = train_test_split(df, test_size=0.3, stratify=df['insurance_benefits_received'], random_state=12345)

# menentukan variabel feature 
X_train = train[feature_names]
X_test = test[feature_names]
y_train = train['insurance_benefits_received']
y_test = test['insurance_benefits_received']
```


```python
# melatih model knn
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(n_neighbors=3)




```python
# Memprediksi menggunakan model knn
knn.predict(X_test).sum()
```




    68




```python
# Metrik Evalusi knn
eval_classifier(y_test, knn.predict(X_test))
```

    F1: 0.40
    Confusion Matrix
    [[0.87333333 0.014     ]
     [0.08133333 0.03133333]]



```python
# melatih model menggunakan data tidak bersakla dan K dari 1 sampai 10 
k = np.arange(1,11)
for k in k:
    print('F1 Skor dan confusion matrix k:', k)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    eval_classifier(y_test, knn.predict(X_test))
    print()
```

    F1 Skor dan confusion matrix k: 1
    F1: 0.59
    Confusion Matrix
    [[0.87266667 0.01466667]
     [0.06       0.05266667]]
    
    F1 Skor dan confusion matrix k: 2
    F1: 0.39
    Confusion Matrix
    [[0.88266667 0.00466667]
     [0.084      0.02866667]]
    
    F1 Skor dan confusion matrix k: 3
    F1: 0.40
    Confusion Matrix
    [[0.87333333 0.014     ]
     [0.08133333 0.03133333]]
    
    F1 Skor dan confusion matrix k: 4
    F1: 0.16
    Confusion Matrix
    [[0.88333333 0.004     ]
     [0.10266667 0.01      ]]
    
    F1 Skor dan confusion matrix k: 5
    F1: 0.15
    Confusion Matrix
    [[0.88       0.00733333]
     [0.10266667 0.01      ]]
    
    F1 Skor dan confusion matrix k: 6
    F1: 0.06
    Confusion Matrix
    [[0.88733333 0.        ]
     [0.10933333 0.00333333]]
    
    F1 Skor dan confusion matrix k: 7
    F1: 0.07
    Confusion Matrix
    [[8.86666667e-01 6.66666667e-04]
     [1.08666667e-01 4.00000000e-03]]
    
    F1 Skor dan confusion matrix k: 8
    F1: 0.05
    Confusion Matrix
    [[0.88733333 0.        ]
     [0.11       0.00266667]]
    
    F1 Skor dan confusion matrix k: 9
    F1: 0.05
    Confusion Matrix
    [[0.88733333 0.        ]
     [0.11       0.00266667]]
    
    F1 Skor dan confusion matrix k: 10
    F1: 0.00
    Confusion Matrix
    [[0.88733333 0.        ]
     [0.11266667 0.        ]]
    


Dari perhitungan di atas, kita dapat melihat bahwa k mempengaruhi metrik evaluasi. Semakin tinggi nilai k semakin rendah skor F1 yang dihasilkan yakni 0.59, Nilai skor F1 tertinggi diperoleh dengan menggunakan nilai k sebesar 1.

Mengukur metrik F1 pada data berskala menggunakan klasifikasi Based-KNN


```python
# Menghitung variabel target untuk penskalaan data

df_scaled['insurance_benefits_received'] = (df_scaled['insurance_benefits'] > 0).astype(int)
df_scaled.head()
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
      <th>gender</th>
      <th>age</th>
      <th>income</th>
      <th>family_members</th>
      <th>insurance_benefits</th>
      <th>insurance_benefits_received</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.630769</td>
      <td>0.627848</td>
      <td>0.166667</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.707692</td>
      <td>0.481013</td>
      <td>0.166667</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.446154</td>
      <td>0.265823</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.323077</td>
      <td>0.527848</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.430769</td>
      <td>0.330380</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# variabel feature dan target untuk data tidak berskala

X = df[feature_names]
y = df['insurance_benefits_received']

# menentukan variabel feature untuk data yang berskala
X_scaled = df_scaled[feature_names]

# memisahkan data menjadi train dan test set
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = train_test_split(
    X, y, X_scaled, test_size = 0.3, random_state=12345
)
```


```python
# model performance using scaled data and k from 1 to 10
k = np.arange(1, 11)
for k in k:
    print('F1 score and confusion matrix for k:', k)
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    eval_classifier(y_test, y_pred)
    print()
```

    F1 score and confusion matrix for k: 1
    F1: 0.97
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.00466667 0.104     ]]
    
    F1 score and confusion matrix for k: 2
    F1: 0.93
    Confusion Matrix
    [[8.90666667e-01 6.66666667e-04]
     [1.40000000e-02 9.46666667e-02]]
    
    F1 score and confusion matrix for k: 3
    F1: 0.95
    Confusion Matrix
    [[0.88933333 0.002     ]
     [0.00866667 0.1       ]]
    
    F1 score and confusion matrix for k: 4
    F1: 0.91
    Confusion Matrix
    [[0.88933333 0.002     ]
     [0.01666667 0.092     ]]
    
    F1 score and confusion matrix for k: 5
    F1: 0.92
    Confusion Matrix
    [[0.88666667 0.00466667]
     [0.01133333 0.09733333]]
    
    F1 score and confusion matrix for k: 6
    F1: 0.90
    Confusion Matrix
    [[0.89       0.00133333]
     [0.018      0.09066667]]
    
    F1 score and confusion matrix for k: 7
    F1: 0.92
    Confusion Matrix
    [[0.88733333 0.004     ]
     [0.01266667 0.096     ]]
    
    F1 score and confusion matrix for k: 8
    F1: 0.90
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.01733333 0.09133333]]
    
    F1 score and confusion matrix for k: 9
    F1: 0.92
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.01466667 0.094     ]]
    
    F1 score and confusion matrix for k: 10
    F1: 0.88
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.02133333 0.08733333]]
    



```python
# melatih model knn menggunakan data berskala dan k dari 1 sampai 10

k = np.arange(1, 11)
for k in k:
    print('F1 Skor dan confusion matrix untuk k:', k)
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    eval_classifier(y_test, y_pred)
    print()
```

    F1 Skor dan confusion matrix untuk k: 1
    F1: 0.97
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.00466667 0.104     ]]
    
    F1 Skor dan confusion matrix untuk k: 2
    F1: 0.93
    Confusion Matrix
    [[8.90666667e-01 6.66666667e-04]
     [1.40000000e-02 9.46666667e-02]]
    
    F1 Skor dan confusion matrix untuk k: 3
    F1: 0.95
    Confusion Matrix
    [[0.88933333 0.002     ]
     [0.00866667 0.1       ]]
    
    F1 Skor dan confusion matrix untuk k: 4
    F1: 0.91
    Confusion Matrix
    [[0.88933333 0.002     ]
     [0.01666667 0.092     ]]
    
    F1 Skor dan confusion matrix untuk k: 5
    F1: 0.92
    Confusion Matrix
    [[0.88666667 0.00466667]
     [0.01133333 0.09733333]]
    
    F1 Skor dan confusion matrix untuk k: 6
    F1: 0.90
    Confusion Matrix
    [[0.89       0.00133333]
     [0.018      0.09066667]]
    
    F1 Skor dan confusion matrix untuk k: 7
    F1: 0.92
    Confusion Matrix
    [[0.88733333 0.004     ]
     [0.01266667 0.096     ]]
    
    F1 Skor dan confusion matrix untuk k: 8
    F1: 0.90
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.01733333 0.09133333]]
    
    F1 Skor dan confusion matrix untuk k: 9
    F1: 0.92
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.01466667 0.094     ]]
    
    F1 Skor dan confusion matrix untuk k: 10
    F1: 0.88
    Confusion Matrix
    [[0.88866667 0.00266667]
     [0.02133333 0.08733333]]
    


Setelah melatih performa model, dapat diamati bahwa skor F1 menurun dari $k$: 1 sebesar 0.97 menjadi 0.88 pada $k$: 10. dapat di simpulkan bahwa meningkatkan nilai k dapat menurunkan skor F1.

Probabilitas klien membayar manfaat asuransi menggunakan model dummy.



```python
for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'Probabilitasnya: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, y_test.shape[0]) # < kode program di sini > 
        
    eval_classifier(y_test, y_pred_rnd)
    
    print()
```

    Probabilitasnya: 0.00
    F1: 0.00
    Confusion Matrix
    [[0.89133333 0.        ]
     [0.10866667 0.        ]]
    
    Probabilitasnya: 0.11
    F1: 0.11
    Confusion Matrix
    [[0.78666667 0.10466667]
     [0.096      0.01266667]]
    
    Probabilitasnya: 0.50
    F1: 0.16
    Confusion Matrix
    [[0.44533333 0.446     ]
     [0.06       0.04866667]]
    
    Probabilitasnya: 1.00
    F1: 0.20
    Confusion Matrix
    [[0.         0.89133333]
     [0.         0.10866667]]
    


Dummy model memberikan kita nilai F1-skor tertinggi 0.2 dengan nilai probabilitas nya 1. model ini memprediksi klien akan membayar manfaat asuransi.  

# Tugas 3. Regresi (dengan Regresi Linear)

Dengan `insurance_benefit` sebagai target, coba evaluasi apa RMSE untuk model regresi linearnya.

Buat implementasi LR. Ingatlah bagaimana solusi tugas regresi linear dirumuskan dalam istilah LA. Periksa RMSE untuk data asli maupun data yang ada skalanya. Bisakah melihat perbedaan RMSE di kedua kasus ini?

Tunjukkan
- $X$ — matriks fitur, satu baris merepresentasikan satu kasus, tiap kolom adalah fitur, kolom pertama terdiri dari kesatuan
- $y$ — target (vektor)
- $\hat{y}$ — estimasi target (vektor)
- $w$ — bobot vektor

Matriks untuk regresi linear dapat dirumuskan sebagai

$$
y = Xw
$$

Tujuan pelatihan untuk menemukan $w$ yang akan meminimalkan jarak L2 (MSE) antara $Xw$ dan $y$:

$$
\min_w d_2(Xw, y) \quad \text{atau} \quad \min_w \text{MSE}(Xw, y)
$$

Ada solusi analitis untuk hal di atas:

$$
w = (X^T X)^{-1} X^T y
$$

Rumus di atas bisa digunakan untuk menemukan bobot $w$ dan yang terakhir dapat digunakan untuk menghitung nilai prediksi

$$
\hat{y} = X_{val}w
$$

Pisahkan keseluruhan data menjadi **training** dan **validation set** dengan proporsi 70:30. Gunakan metrik RMSE untuk evaluasi model.


```python
class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # tambahkan satuan
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)# < kode program di sini >

    def predict(self, X):
        
        # tambahkan satuan
        X2 = np.append(np.ones([len(X),1]), X, axis=1)# < kode program di sini >
        y_pred = X2.dot(self.weights)# < kode program di sini >
        
        return y_pred
```


```python
def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    
```

Menetukan Variabel feature (X) dan target (y) pada Dataset dan memisahkan train, valid dan test set. dan melatih model tanpa penskalaan data.


```python
X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)
```

    [-9.43539012e-01  3.57495491e-02  1.64272726e-02 -2.60743659e-07
     -1.16902127e-02]
    RMSE: 0.34
    R2: 0.66


Menetukan Variabel feature (X) dan target (y) pada Dataset dan memisahkan train, valid dan test set. dan melatih model menggunakan penskalaan data.


```python
X = df_scaled[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df_scaled['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)
```

    [-0.94353901  2.32372069  0.01642727 -0.02059875 -0.07014128]
    RMSE: 0.34
    R2: 0.66


`kesimpulan`:

Setelah melakukan implementasi pada model regresi linear dan menghitung RMSE dan R2 skor koefisisen determinasi pada dataset asli dan data yang dilakukan penskalaan. kita dapat melihat bahwa tidak ada perbedaan dalam skor RMSE dan R2. sehingga dapat disimpulkan bahwa dalam evaluasi metric yang dihasilkan model ini memberikan hasil yang sama. 

# Tugas 4. Pengaburan Data

Cara terbaik untuk mengaburkan data adalah dengan mengalikan fitur-fitur numerik (ingat bahwa fitur-fitur tersebut bisa di lihat di matriks $X$) dengan matriks yang dapat dibalik $P$. 

$$
X' = X \times P
$$

Coba lakukan hal itu dan periksa hasil nilai-nilai fiturnya setelah ditransformasikan. Properti invertible penting di sini, jadi pastikan $P$ dapat dibalik.

Mungkin kamu ingin meninjau kembali pelajaran 'Matriks dan Operasi Matriks -> Perkalian Matriks' untuk mengingat aturan perkalian matriks dan implementasinya dengan NumPy.


```python
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]
```


```python
X = df_pn.to_numpy()
X
```




    array([[1.00e+00, 4.10e+01, 4.96e+04, 1.00e+00],
           [0.00e+00, 4.60e+01, 3.80e+04, 1.00e+00],
           [0.00e+00, 2.90e+01, 2.10e+04, 0.00e+00],
           ...,
           [0.00e+00, 2.00e+01, 3.39e+04, 2.00e+00],
           [1.00e+00, 2.20e+01, 3.27e+04, 3.00e+00],
           [1.00e+00, 2.80e+01, 4.06e+04, 1.00e+00]])



Membuat matriks acak $P$.


```python
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))
P

```




    array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
           [0.09417735, 0.97562235, 0.7611397 , 0.78606431],
           [0.12811363, 0.45038594, 0.37079802, 0.92676499],
           [0.64386512, 0.82276161, 0.4434142 , 0.22723872]])



Periksa apakah matriks $P$ bisa dibalik


```python
# memeriksa apakah matriks P dapat dibalik
np.matmul(P, np.linalg.inv(P))
```




    array([[ 1.00000000e+00, -1.69848573e-16, -7.58122972e-17,
            -1.13112497e-16],
           [-6.94895396e-17,  1.00000000e+00, -7.10568689e-17,
             3.59096970e-17],
           [-1.21269339e-16, -8.01461326e-17,  1.00000000e+00,
             4.30764008e-19],
           [-3.60694539e-16, -5.55430227e-16,  3.08072404e-16,
             1.00000000e+00]])



Kita dapat melihat bahwa matriks $P$ dapat dibalik karena ketika kita mengkalikan $P$ dengan inversnya $P-1$ 
Kita melihat bahwa matriks P dapat dibalik karena ketika kita mengalikan P dengan inversnya P
, tidak terjadi kesalahan pada hasil invers.

Bisakah kamu menebak usia klien atau pendapatan setelah melakukan transformasi?


```python
# merubah data X 
X_transform = np.matmul(X, P)
X_transform
```




    array([[ 6359.71527314, 22380.40467609, 18424.09074184, 46000.69669016],
           [ 4873.29406479, 17160.36702982, 14125.78076133, 35253.45577301],
           [ 2693.11742928,  9486.397744  ,  7808.83156024, 19484.86063067],
           ...,
           [ 4346.2234249 , 15289.24126492, 12586.16264392, 31433.50888552],
           [ 4194.09324155, 14751.9910242 , 12144.02930637, 30323.88763426],
           [ 5205.46827354, 18314.24814446, 15077.01370762, 37649.59295455]])



Bisakah kamu memulihkan data asli dari $X'$ jika sudah tahu $P$-nya? Coba periksa perhitungan dengan memindahkan $P$ dari sisi kanan atas rumus, ke bagian kiri. Aturan perkalian matriks sangat berguna di sini.


```python
# Memulihakan data asli dari X
X_inv = X_transform.dot(np.linalg.inv(P))
X_inv
```




    array([[ 1.00000000e+00,  4.10000000e+01,  4.96000000e+04,
             1.00000000e+00],
           [-4.47363596e-12,  4.60000000e+01,  3.80000000e+04,
             1.00000000e+00],
           [-2.51586878e-12,  2.90000000e+01,  2.10000000e+04,
             9.52452315e-13],
           ...,
           [-1.92837871e-12,  2.00000000e+01,  3.39000000e+04,
             2.00000000e+00],
           [ 1.00000000e+00,  2.20000000e+01,  3.27000000e+04,
             3.00000000e+00],
           [ 1.00000000e+00,  2.80000000e+01,  4.06000000e+04,
             1.00000000e+00]])



Tampilkan ketiga kasus untuk beberapa klien
- Data asli
- Data yang sudah ditransformasikan
- Data yang telah dikembalikan ke semula


```python
print('Data Asli')
print(X[:3])
print()
print('Data yang sudah ditransformasi')
print(X_transform[:3])
print()
print('Data yang telah di inverse')
print(X_inv[:3])
```

    Data Asli
    [[1.00e+00 4.10e+01 4.96e+04 1.00e+00]
     [0.00e+00 4.60e+01 3.80e+04 1.00e+00]
     [0.00e+00 2.90e+01 2.10e+04 0.00e+00]]
    
    Data yang sudah ditransformasi
    [[ 6359.71527314 22380.40467609 18424.09074184 46000.69669016]
     [ 4873.29406479 17160.36702982 14125.78076133 35253.45577301]
     [ 2693.11742928  9486.397744    7808.83156024 19484.86063067]]
    
    Data yang telah di inverse
    [[ 1.00000000e+00  4.10000000e+01  4.96000000e+04  1.00000000e+00]
     [-4.47363596e-12  4.60000000e+01  3.80000000e+04  1.00000000e+00]
     [-2.51586878e-12  2.90000000e+01  2.10000000e+04  9.52452315e-13]]


Mungkin kamu bisa melihat ada beberapa nilai yang tidak benar-benar sama dengan data sebelum ditransformasi. Kenapa bisa begitu?

Jawaban :
Beberapa nilai tidak sama dengan yang ada di data asli karena transformasi dan presisi perhitungan NumPy dari data mendekati nol.

##  Buktikan bahwa pengaburan data bisa bekerja dengan LR

Tugas regresi telah diselesaikan dengan regresi linear di sini. Tugas selanjutnya adalah untuk membuktikan analytically bahwa metode pengaburan data tertentu tidak akan memengaruhi prediksi nilai regresi linear, yaitu nilai-nilai prediksi tersebut tidak akan berubah dari hasil awalnya. Apakah kamu yakin? Kamu harus membuktikannya!

Jadi, data dikaburkan dan ada $X \times P$ sekarang $X$. Akibatnya, ada bobot lain $w_P$ 
$$
w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
$$

Bagaimana menghubungkan $w$ dan $w_P$ jika menyederhanakan rumus untuk $w_P$ di atas? 

Nilai apa yang akan $w_P$ prediksi? 

Apa artinya kualitas regresi linier jika mengukurnya dengan RMSE?

Periksa properti apendiks B matriks di akhir *notebook*. Ada beberapa rumus di sana!

Tidak ada kode yang begitu penting di sesi ini, hanya penjelasan analitis!

**Jawaban**

Penjelasan analisis:





1. Bagaimana menghubungkan  𝑤 dan  𝑤𝑃 jika menyederhanakan rumus untuk  𝑤𝑃 di atas?


Untuk menghubungkan rumus 

$$ w_P = [(XP)^T XP]^{-1} (XP)^T y $$

Kita perlu memperluas $ XP^T $ menggunakan reversivitas properti transpose 


$$ w_p = [P^T X^T XP]^{-1} P^T X^T y$$

lalu kita akan mengatur ulang rumus : 

 $$ w_p = (P^T (X^T X) P)^{-1} P^T X^T y $$

dan mengubah $P^T$ menjadi $P^{-1}$ contoh : $(P^T (X^T X) P)^{-1}$ :

$$ w_p = P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T y $$

Karena Jika matriks $A$ apapun dikali dengan matriks identitas (atau sebaliknya), hasilnya adalah matriks $A$ itu sendiri. $ A⋅E=E⋅A=A $ an dapat dirumuskan  $(P^T)^{-1} P^T = P^T (P^T)^{-1}  = I$

Dengan persamaan : $$ w_p = P^{-1} (X^T X)^{-1} I X^T y $$

Lalu mengubah $w = (X^T X)^{-1} X^T y$ kedalam persamaan :

$$\therefore w_p = P^{-1} w $$

2. Nilai apa yang akan  𝑤𝑃 prediksi?

Mengingat bahwa :  $a = Xw$

menyerupai persamaan : $a' = X'w_p$

dimana jika dengan contoh kasus diatas : $X' = XP$

jika kita masukkan dalam persamaan $w_p = P^{-1}w$

dan mengganti nilai $a'$ sehingga kita mempunyai persamaan :

$$\begin{align*}
    a' &= XP.P^{-1}w = XIw \\
    \therefore a' &= Xw = a
\end{align*}$$

3. Apa artinya kualitas regresi linier jika mengukurnya dengan RMSE?

Sejak $a'$ sama dengan $a$, kami dapat berasumsi RMSE yang dihitung untuk kumpulan data asli dan dengan penskalaan menjadi sama maka Kualitas model regresi linier yang diukur menggunakan RMSE akan serupa.

## Uji regresi linear dengan pengaburan data

Buktikan bila regresi linier dapat bekerja secara komputasional apabila diterapkan transformasi pengaburan data.

Buat prosedur atau kelas yang menjalankan regresi linier dengan pengaburan data. Kamu bisa menggunakan regresi linier yang tersedia di scikit-learn atau milikmu sendiri.

Jalankan regresi linier untuk data asli dan data yang disamarkan, bandingkan nilai prediksi dan RMSE, nilai metrik $R^2$. Apakah ada perbedaan?

**Prosedur**

- Buat matriks persegi $P$ dari angka acak.
- Periksa apakah bisa dibalik. Jika tidak, ulangi langkah pertama sampai mendapatkan matriks yang bisa dibalik.
- <! komentar di sini !>
- Gunakan $XP$ sebagai matriks fitur baru 


```python
# matrix persegi P dengan angka yang acak
rng = np.random.default_rng(seed=42)
p = rng.random(size=(X.shape[1], X.shape[1]))
p
```




    array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
           [0.09417735, 0.97562235, 0.7611397 , 0.78606431],
           [0.12811363, 0.45038594, 0.37079802, 0.92676499],
           [0.64386512, 0.82276161, 0.4434142 , 0.22723872]])




```python
# memriksa jika matrix P dapar di invers
np.matmul(P, np.linalg.inv(P))
```




    array([[ 1.00000000e+00, -1.69848573e-16, -7.58122972e-17,
            -1.13112497e-16],
           [-6.94895396e-17,  1.00000000e+00, -7.10568689e-17,
             3.59096970e-17],
           [-1.21269339e-16, -8.01461326e-17,  1.00000000e+00,
             4.30764008e-19],
           [-3.60694539e-16, -5.55430227e-16,  3.08072404e-16,
             1.00000000e+00]])




```python
# Menggunakan X P sebagai fitur baru 
personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# feature
X = df_pn.to_numpy()

# target
y = df["insurance_benefits"].to_numpy()
```


```python
# linear regression dengan data asli
# Memisahkan data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Model LR
lr = MyLinearRegression() 
lr.fit(X_train, y_train)
print(lr.weights)

# Membuat prediksi dan metrik evaluasi
y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)
```

    [-9.43539012e-01  1.64272726e-02  3.57495491e-02 -2.60743659e-07
     -1.16902127e-02]
    RMSE: 0.34
    R2: 0.66



```python
# linear Regression dengan pengaburan data
X_transform = X.dot(P)

# Memisahkan data
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_transform, y, test_size=0.3, random_state=12345)
                                                                        

# Model LR
lr = MyLinearRegression()
lr.fit(X_train_trans, y_train_trans)
print(lr.weights)

y_test_pred = lr.predict(X_test_trans)
eval_regressor(y_test_trans, y_test_pred)


```

    [-0.94353902 -0.05791721 -0.01546567  0.09871889 -0.02397536]
    RMSE: 0.34
    R2: 0.66


Kami membuktikan secara matematis bahwa RMSE dan $R^2$ skor untuk data asli dan pengaburan data sama. Sekarang dengan menggunakan cara komputasi, kami memperoleh RMSE serupa sebesar 0,34 dan
 $R^2$ skor 0,66 untuk data asli dan pengaburan data.

# Kesimpulan

1. Persiapan preprosessing data sudah berhasil, dengan informasi dataset 5000 baris dan 5 kolom, untuk kemudahan kita juga mengubah tipe data pada kolom age dari float ke tipe interger. kita juga memeriksa tida ada nilai yang hilang pada data. sekarang kita dapat mengeksplor data menggunakan exploratory data analysis.
------
2. Penskalaan fitur penting untuk algoritme kNN karena menghitung jarak antar data. Data yang tidak diskalakan memepengaruhi algoritma kNN karena algoritma lebih mementingkan fitur tertentu dengan rentang angka yang lebih tinggi daripada yang lain. Dalam latihan yang menggunakan jarak euclidean misalnya, metrik jarak yang dihitung untuk data yang tidak diskalakan berkisar antara 0,00 hingga 5,09, sedangkan untuk data yang diskalakan, kami memperoleh rentang metrik jarak dari 0,00 hingga 0,03. Ini menunjukkan dampak penskalaan pada data. kNN dengan jarak euclidean peka terhadap besaran, oleh karena itu, data harus diskalakan agar semua fitur memiliki bobot yang sama. Jarak terdekat yang dihasilkan pada metrik euclidian dan manhattan metrik terlepas dari penskalaannya hampir serupa pada index ke dua yakni dengan nilai 1.0 pada metrik tidak berskala dan 0.006329 pada data berskala. setelah itu jarak yang diihasilkan berbeda.
-----
3. Setelah melatih performa model, dapat diamati bahwa skor F1 menurun dari `k`: 1 sebesar 0.97 menjadi 0.88 pada `k`: 10. dapat di simpulkan bahwa meningkatkan nilai k dapat menurunkan skor F1.
-----
4. Setelah melakukan implementasi pada model regresi linear dan menghitung RMSE dan R2 skor koefisisen determinasi pada dataset asli dan data yang dilakukan penskalaan. kita dapat melihat bahwa tidak ada perbedaan dalam skor RMSE dan R2. sehingga dapat disimpulkan bahwa dalam evaluasi metric yang dihasilkan model ini memberikan hasil yang sama.
-----
5. Kami membuktikan secara matematis bahwa RMSE dan $R^2$ skor untuk data asli dan pengaburan data sama. Sekarang dengan menggunakan cara komputasi, kami memperoleh RMSE serupa sebesar 0,34 dan $R^2$ skor 0,66 untuk data asli dan pengaburan data. Hal ini membuktikan bahwa nilai prediksi, RMSE dan $R^2$  metrik akan sama untuk data asli dan pengaburan data.

# Daftar Periksa

Ketik 'x' untuk memeriksa. Lalu tekan Shitf+Enter

- [x]  Jupyter Notebook dibuka
- [ ]  Tidak ada kode yang salah
- [ ]  Sel disusun sesuai urutan yang logis
- [ ]  Tugas 1 telah dikerjakan
    - [ ]  Ada prosedur yang bisa menampilkan k klien yang mirip dengan klien tertentu
    - [ ]  Prosedur diuji untuk keempat kombinasi yang diusulkan
    - [ ]  Pertanyaan terkait skala/jarak sudah terjawab
- [ ]  Tugas 2 telah dikerjakan
    - [ ]  Telah dibuat model klasifikasi acak untuk semua level probabilitas
    - [ ]  Model klasifikasi kNN telah dibuat, baik untuk data asli dan yang telah diberi skala, metrik F1 telah diperhitungkan.
- [ ]  Tugas 3 telah dikerjakan
    - [ ]  Solusi regresi linear diimplementasikan menggunakan pengoperasian matriks.
    - [ ]  RMSE telah dihitung untuk solusi yang diimplementasikan.
- [ ]  Tugas 4 telah dikerjakan
    - [ ]  Data dikaburkan dengan acak dan matriks P yang bisa dibalik
    - [ ]  Data yang dikaburkan telah dipulihkan, ada beberapa contoh yang ditampilkan
    - [ ]  Terdapat bukti analitis bahwa transformasi tidak mempengaruhi RMSE
    - [ ]  Terdapat bukti perhitungan bahwa transformasi tidak memengaruhi RMSE
- [ ]  Ada kesimpulan

# Apendiks 

## Apendiks A: Menulis Rumus di Jupyter Notebooks

Kamu dapat menulis rumus di Jupyter Notebook dalam bahasa markup yang disediakan oleh sistem penerbitan berkualitas tinggi yang disebut $\LaTeX$ (diucapkan "Lah-tech"), dan rumus tersebut akan terlihat seperti rumus yang ada di buku teks.

Untuk memasukkan rumus ke dalam teks, letakkan tanda dolar (\\$) sebelum dan sesudah teks rumus. $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.

Jika rumus harus dalam satu paragraf tersendiri, letakkan dua tanda dolar (\\$\\$) sebelum dan sesudah teks.

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
$$

Bahasa markup [LaTeX](https://en.wikipedia.org/wiki/LaTeX) sangat populer di kalangan orang-orang yang menggunakan rumus dalam artikel, buku, dan teks. Bahasa tesebut bisa saja tampak rumit, tetapi sebenarnya dasar bahasa markup itu mudah. Periksa dua halaman [cheatsheet](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) berikut untuk mempelajari cara membuat rumus yang paling umum.

## Apendiks B: Properti Matriks

Matriks memiliki banyak properti di Aljabar Linear. Beberapa di antaranya ada di daftar berikut yang bisa membantu dengan bukti analisis di tugas ini.

<table>
<tr>
<td>Distribusi</td><td>$A(B+C)=AB+AC$</td>
</tr>
<tr>
<td>Non-komutatif</td><td>$AB \neq BA$</td>
</tr>
<tr>
<td>Sifat asosiatif perkalian</td><td>$(AB)C = A(BC)$</td>
</tr>
<tr>
<td>Properti identitas perkalian</td><td>$IA = AI = A$</td>
</tr>
<tr>
<td></td><td>$A^{-1}A = AA^{-1} = I$
</td>
</tr>    
<tr>
<td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
</tr>    
<tr>
<td>Kebalikan dari transpos produk matriks,</td><td>$(AB)^T = B^TA^T$</td>
</tr>    
</table>


```python

```
