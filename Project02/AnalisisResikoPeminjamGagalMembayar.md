# Menganalisis risiko peminjam gagal membayar

Proyek Anda ialah menyiapkan laporan untuk bank bagian kredit. Anda harus mencari tahu pengaruh status perkawinan seorang nasabah dan jumlah anak terhadap probabilitas ketepatan waktu dalam melunasi pinjaman. Bank sudah memiliki beberapa data mengenai kelayakan kredit nasabah.

Laporan Anda akan dipertimbangkan pada saat membuat **penilaian kredit** untuk calon nasabah. **Penilaian kredit** digunakan untuk mengevaluasi kemampuan calon peminjam untuk melunasi pinjaman mereka.

Sebelum melakukan analisis data pada resiko peminjam gagal membayar. langkah yang harus dilakukan adalah :

1. Mengimport Libraries dan membuka file serta membaca informasi pada dataset secara umum. 

2. Explorasi Data / Data Quality Checking yang meliputi :
     
     Melakukan analisis data secara global dengan mengamati dengan detail value dari semua kolom serta tipe datanya. lalu memeriksa kualitas dari data tersebut. 
a. Memeriksa value yang unique pada datasets apakah terdapat gaya penulisan yang tidak sesuai dengan formatnya.
b. Memeriksa value duplikasi data pada datasets. 
c. Memeriksa value apakah terdapat data yang tidak wajar sesuai dengan karakteristik pada value column tersebut. 
d. Memfilter value pada column yang terindikasi nilai yang hilang apakah terdapat nilai yang hilang secara acak atau terpola
   
3. Data Cleansing / Membersihkan data 
    Melakukan data cleansing yang terindetifikasi :
a. memperbaiki value yang terduplikasi.
b. memperbaiki value yang unique pada gaya penulisan.
c. memperbaiki value yang teridentifikasi tidak wajar.
d. memperbaiki nilai yang hilang.  

4. Data Categorizing / Pengkategorian data 
     Melakukan pengkategorian data, untuk menjawab hipotesis. 
     
5. Hipotesis / Kesimpulan akhir dari pertanyaan.

## 1. Buka *file* data dan baca informasi umumnya. 





```python
# Memuat semua perpustakaan
import pandas as pd, numpy as np


```


```python
# muat data
try:
    df = pd.read_csv('/datasets/credit_scoring_eng.csv')
    
except:
    df = pd.read_csv('credit_scoring_eng.csv')
```

## 2. Soal 1. Eksplorasi Data

**Deskripsi Data**
- *`children`* - jumlah anak dalam keluarga
- *`days_employed`* - pengalaman kerja dalam hari
- *`dob_years`* - usia klien dalam tahun
- *`education`* - pendidikan klien
- *`education_id`* - tanda pengenal pendidikan
- *`family_status`* - status perkawinan
- *`family_status_id`* - tanda pengenal status perkawinan
- *`gender`* - jenis kelamin klien
- *`income_type`* - jenis pekerjaan
- *`debt`* - apakah klien memiliki hutang pembayaran pinjaman
- *`total_income`* - pendapatan bulanan
- *`purpose`* - tujuan mendapatkan pinjaman

Pada langkah ini kita akan melihat seputar informasi umum pada :
a. Berapa banyak baris dan column pada datasets



```python
# Mari kita lihat berapa banyak baris dan kolom yang dimiliki oleh dataset kita
print(f'panjang baris pada dataset dan jumlah column pada dataset adalah : {df.shape}')
```

    panjang baris pada dataset dan jumlah column pada dataset adalah : (21525, 12)



```python
print(f'detail column yang terdapat pada datasets sebagai berikut :')
df.columns
```

    detail column yang terdapat pada datasets sebagai berikut :





    Index(['children', 'days_employed', 'dob_years', 'education', 'education_id',
           'family_status', 'family_status_id', 'gender', 'income_type', 'debt',
           'total_income', 'purpose'],
          dtype='object')




```python
# mari menampilkan N baris pertama
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-8437.673028</td>
      <td>42</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>40620.102</td>
      <td>purchase of the house</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-4024.803754</td>
      <td>36</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>17932.802</td>
      <td>car purchase</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-5623.422610</td>
      <td>33</td>
      <td>Secondary Education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>23341.752</td>
      <td>purchase of the house</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-4124.747207</td>
      <td>32</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>42820.568</td>
      <td>supplementary education</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>340266.072047</td>
      <td>53</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>25378.572</td>
      <td>to have a wedding</td>
    </tr>
  </tbody>
</table>
</div>



**Setelah menampilkan baris dan kolom serta baris pertama dari sampel data, perlu penyelidikan lebih lanjut dengan alasan :**
**1. kurangnya informasi lebih dalam mengenai sampel data.**
**2. adanya gaya penulisan yang tidak sesuai.**
**3. memastikan ada atau tidaknya data terduplikasi dan data dengan value yang hilang.**
***4. terdapat nilai yang tidak wajar pada value days employed***


```python
# Mendapatkan informasi data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21525 entries, 0 to 21524
    Data columns (total 12 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   children          21525 non-null  int64  
     1   days_employed     19351 non-null  float64
     2   dob_years         21525 non-null  int64  
     3   education         21525 non-null  object 
     4   education_id      21525 non-null  int64  
     5   family_status     21525 non-null  object 
     6   family_status_id  21525 non-null  int64  
     7   gender            21525 non-null  object 
     8   income_type       21525 non-null  object 
     9   debt              21525 non-null  int64  
     10  total_income      19351 non-null  float64
     11  purpose           21525 non-null  object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 2.0+ MB


**dengan memanggil informasi dari semua column terdapat indetifikasi sementara nilai yang hilang di days_employed & total income dengan jumlah row yang berbeda pada kedua column**


```python
# Mari kita melihat tabel yang telah difilter dengan nilai yang hilang di kolom pertama dengan data yang hilang
filter_df = df.loc[(df['days_employed'].isna()) & df['total_income'].isna()]
filter_df.head()
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>NaN</td>
      <td>65</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>M</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>to have a wedding</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>NaN</td>
      <td>41</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>civil servant</td>
      <td>0</td>
      <td>NaN</td>
      <td>education</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>NaN</td>
      <td>63</td>
      <td>secondary education</td>
      <td>1</td>
      <td>unmarried</td>
      <td>4</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>building a real estate</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>NaN</td>
      <td>50</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>civil servant</td>
      <td>0</td>
      <td>NaN</td>
      <td>second-hand car purchase</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>NaN</td>
      <td>54</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>retiree</td>
      <td>1</td>
      <td>NaN</td>
      <td>to have a wedding</td>
    </tr>
  </tbody>
</table>
</div>



Terdapat Value Nan pada column days employed dan total income. 


```python
# Menampilkan jumlah column dengan nilai yang hilang pada setiap column
df.isna().sum()
```




    children               0
    days_employed       2174
    dob_years              0
    education              0
    education_id           0
    family_status          0
    family_status_id       0
    gender                 0
    income_type            0
    debt                   0
    total_income        2174
    purpose                0
    dtype: int64



Terdapat 2174 value yang terindikasi hilang pada column days employed dan total income.


**Terdapat nilai yang hilang pada column days_employed & dob_years dengan pola yang simetris yakni jumlah value yang hilang pada kedua column. sebaiknya kita identifikasi lebih dalam untuk memastikan***


```python
# Mari kita menerapkan beberapa persyaratan untuk memfilter data dan melihat jumlah baris dalam tabel yang difilter.
test  = pd.DataFrame(df.isna().sum(), columns=['missing_values'])
```

**Kesimpulan menengah**

[Apakah jumlah baris dalam tabel yang difilter sesuai dengan jumlah nilai yang hilang? Kesimpulan apa yang bisa kita buat dari hal ini?]

**jumlah baris dalam tabel yg difilter menggunakan isna.count() memiliki value yang sama, namun dengan menggunakan isna.sum() ditemukan nilai yang hilang sebesar 2174 row. dapat disimpulkan bahwa terdapat nilai yang hilang pada column['days_employed,total_income] mari kita selidiki lebih lanjut**  

[Hitung persentase nilai yang hilang yang dibandingkan dengan seluruh kumpulan data. Apakah ini merupakan bagian data yang sangat besar? Jika demikian, Anda mungkin ingin mengisi nilai yang hilang. Untuk melakukannya, pertama-tama kita harus mempertimbangkan apakah data yang hilang bisa jadi disebabkan oleh karakteristik klien tertentu, seperti jenis pekerjaan atau yang lainnya. Anda perlu memutuskan karakteristik mana yang *Anda* pikir mungkin menjadi alasannya. Kedua, kita harus memeriksa apakah terdapat ketergantungan nilai yang hilang pada nilai indikator lain dengan kolom yang berisi karakteristik klien spesifik yang teridentifikasi.]

**setelah menghitung persentasa antara nilai column yang hilang dengan jumlah range dataset didapat 10.1% nilai yang hilang pada column (days_employed & total_income).**

[Jelaskan langkah Anda selanjutnya dan bagaimana hubungannya dengan kesimpulan yang Anda buat sejauh ini.]

**langkah selanjutnya adalah memeriksa data pada column yang hilang identifikasi nilai yang hilang secara detail apakah ada pola yang terhubung dengan satu sama lain.**


```python
# Mari kita memeriksa klien yang tidak memiliki data tentang karakteristik yang teridentifikasi dan kolom dengan nilai yang hilang
test['percent'] = test['missing_values'] / len(df)
test
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
      <th>missing_values</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>children</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>days_employed</th>
      <td>2174</td>
      <td>0.100999</td>
    </tr>
    <tr>
      <th>dob_years</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>education</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>education_id</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>family_status</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>family_status_id</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>income_type</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>debt</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_income</th>
      <td>2174</td>
      <td>0.100999</td>
    </tr>
    <tr>
      <th>purpose</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Memeriksa distribusi 
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21525 entries, 0 to 21524
    Data columns (total 12 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   children          21525 non-null  int64  
     1   days_employed     19351 non-null  float64
     2   dob_years         21525 non-null  int64  
     3   education         21525 non-null  object 
     4   education_id      21525 non-null  int64  
     5   family_status     21525 non-null  object 
     6   family_status_id  21525 non-null  int64  
     7   gender            21525 non-null  object 
     8   income_type       21525 non-null  object 
     9   debt              21525 non-null  int64  
     10  total_income      19351 non-null  float64
     11  purpose           21525 non-null  object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 2.0+ MB


[Deksripsikan yang Anda temukan di sini.]

Deskripsi pada data distribusi awal dan distribusi seluruh dataset memiliki karakteristik value yang sama yaitu :
1. pada column children terdapat value yang tidak wajar yaitu -1 dan 20.
2. pada colum days employed terdapat value dengan nilai negatif.
3. pada column dob_years terdapat value yang menunjukkan umur nasabah 0th.
4. pada column education terdapat gaya penulisan yang berbeda dan tidak beraturan untuk value dengan defenisi yang sama.
5. pada column purpose terdapat beberapa 'str' yang mendefenisikan arti yang sama.
6. pada column gender terdapat value yang tidak diketahu gendernya yakni XNA
7. pada column purpose terdapat value bawaan dengan defenisi yang sama.

[Ajukan ide-ide Anda tentang mengapa menurut Anda nilai-nilai tersebut kemungkinan hilang. Apakah menurut Anda mereka hilang secara acak atau terdapat pola?]

Belum dapat dipastikan kemungkinan nilai-nilai yang hilang tersebut, hilang secara acak atau terdapat pola 

[Mari kita mulai memeriksa apakah nilai hilang secara acak.]


```python
# Memeriksa distribusi pada keseluruhan dataset
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education_id</th>
      <th>family_status_id</th>
      <th>debt</th>
      <th>total_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>21525.000000</td>
      <td>19351.000000</td>
      <td>21525.000000</td>
      <td>21525.000000</td>
      <td>21525.000000</td>
      <td>21525.000000</td>
      <td>19351.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.538908</td>
      <td>63046.497661</td>
      <td>43.293380</td>
      <td>0.817236</td>
      <td>0.972544</td>
      <td>0.080883</td>
      <td>26787.568355</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.381587</td>
      <td>140827.311974</td>
      <td>12.574584</td>
      <td>0.548138</td>
      <td>1.420324</td>
      <td>0.272661</td>
      <td>16475.450632</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.000000</td>
      <td>-18388.949901</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3306.762000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>-2747.423625</td>
      <td>33.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16488.504500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>-1203.369529</td>
      <td>42.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>23202.870000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>-291.095954</td>
      <td>53.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>32549.611000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20.000000</td>
      <td>401755.400475</td>
      <td>75.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>362496.645000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Menampilkan nilai yang unik pada column children
df['children'].unique()
```




    array([ 1,  0,  3,  2, -1,  4, 20,  5])




```python
# Menampilkan nilai yang unik pada column days employed
df['days_employed'].unique()
```




    array([-8437.67302776, -4024.80375385, -5623.42261023, ...,
           -2113.3468877 , -3112.4817052 , -1984.50758853])




```python
# Menampilkan nilai yang unik pada column dob years
df['dob_years'].unique()
```




    array([42, 36, 33, 32, 53, 27, 43, 50, 35, 41, 40, 65, 54, 56, 26, 48, 24,
           21, 57, 67, 28, 63, 62, 47, 34, 68, 25, 31, 30, 20, 49, 37, 45, 61,
           64, 44, 52, 46, 23, 38, 39, 51,  0, 59, 29, 60, 55, 58, 71, 22, 73,
           66, 69, 19, 72, 70, 74, 75])




```python
# Menampilkan nilai yang unik pada column education
df['education'].unique()
```




    array(["bachelor's degree", 'secondary education', 'Secondary Education',
           'SECONDARY EDUCATION', "BACHELOR'S DEGREE", 'some college',
           'primary education', "Bachelor's Degree", 'SOME COLLEGE',
           'Some College', 'PRIMARY EDUCATION', 'Primary Education',
           'Graduate Degree', 'GRADUATE DEGREE', 'graduate degree'],
          dtype=object)




```python
# Menampilkan nilai yang unik pada column education id
df['education_id'].unique()
```




    array([0, 1, 2, 3, 4])




```python
# Menampilkan nilai yang unik pada column family status
df['family_status'].unique()
```




    array(['married', 'civil partnership', 'widow / widower', 'divorced',
           'unmarried'], dtype=object)




```python
# Menampilkan nilai yang unik pada column family status id
df['family_status_id'].unique()
```




    array([0, 1, 2, 3, 4])




```python
# Menampilkan nilai yang unik pada column gender
df['gender'].unique()
```




    array(['F', 'M', 'XNA'], dtype=object)




```python
# Menampilkan nilai yang unik pada column income type
df['income_type'].unique()
```




    array(['employee', 'retiree', 'business', 'civil servant', 'unemployed',
           'entrepreneur', 'student', 'paternity / maternity leave'],
          dtype=object)




```python
# Menampilkan nilai yang unik pada column debt
df['debt'].unique()
```




    array([0, 1])




```python
# Menampilkan Nilai yang unik pada total income
df['total_income'].unique()
```




    array([40620.102, 17932.802, 23341.752, ..., 14347.61 , 39054.888,
           13127.587])




```python
# Menampilkan nilai yang unik pada column purpose
df['purpose'].unique()
```




    array(['purchase of the house', 'car purchase', 'supplementary education',
           'to have a wedding', 'housing transactions', 'education',
           'having a wedding', 'purchase of the house for my family',
           'buy real estate', 'buy commercial real estate',
           'buy residential real estate', 'construction of own property',
           'property', 'building a property', 'buying a second-hand car',
           'buying my own car', 'transactions with commercial real estate',
           'building a real estate', 'housing',
           'transactions with my real estate', 'cars', 'to become educated',
           'second-hand car purchase', 'getting an education', 'car',
           'wedding ceremony', 'to get a supplementary education',
           'purchase of my own house', 'real estate transactions',
           'getting higher education', 'to own a car', 'purchase of a car',
           'profile education', 'university education',
           'buying property for renting out', 'to buy a car',
           'housing renovation', 'going to university'], dtype=object)



**Kesimpulan menengah**

[Apakah distribusi dalam dataset yang asli mirip dengan distribusi tabel yang telah difilter? Apa artinya itu untuk kita?]

#### Distribusi pada dataset asli mirip dengan distribusi dataset yang telah difilter. terdapat kesamaan masalah pada dataset yang telah difilter dan dataset yang asli. data belum dapat diambil kesimpulan apapun untuk nilai yang hilang secara acak maupun terpola.

[Jika menurut Anda kita belum dapat membuat kesimpulan apa pun, mari kembali menyelidiki dataset kita lebih lanjut. Mari kita pikirkan alasan lain yang dapat menyebabkan data hilang dan periksa apakah kita dapat menemukan pola yang dapat membuat kita berpikir bahwa nilai yang hilang tidaklah secara acak. Karena ini merupakan pekerjaan Anda, bagian ini adalah bagian opsional.]


```python
#Periksa penyebab dan pola lain yang dapat mengakibatkan nilai yang hilang
df_filter2 = df.loc[(df['income_type'].isna()) & df['total_income'].isna()]
df_filter2
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



**Kesimpulan menengah**

[Apakah pada akhirnya kita dapat memastikan bahwa nilai yang hilang adalah suatu kebetulan? Periksa hal lain yang menurut Anda penting di sini.]

#### Setelah memeriksa penyebab nilai yang hilang dengan menggunakan column income type tidak didapat nilai yang hilang secara kebetulan. mari kita periksa lebih detail pada column income_type dimana terdapat beberapa pekerjaan yang mungkin tidak berpenghasilan. 


```python
# Memeriksa pola lainnya - jelaskan pola tersebut
# # Memeriksa pola lainnya dengan detail income type employee
df.loc[(df['income_type']=='employee') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>82</th>
      <td>2</td>
      <td>NaN</td>
      <td>50</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>housing</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0</td>
      <td>NaN</td>
      <td>52</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>housing</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2</td>
      <td>NaN</td>
      <td>35</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>housing transactions</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>NaN</td>
      <td>44</td>
      <td>SECONDARY EDUCATION</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>buy residential real estate</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0</td>
      <td>NaN</td>
      <td>47</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>profile education</td>
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
      <th>21432</th>
      <td>1</td>
      <td>NaN</td>
      <td>38</td>
      <td>some college</td>
      <td>2</td>
      <td>unmarried</td>
      <td>4</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>housing transactions</td>
    </tr>
    <tr>
      <th>21463</th>
      <td>1</td>
      <td>NaN</td>
      <td>35</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>having a wedding</td>
    </tr>
    <tr>
      <th>21495</th>
      <td>1</td>
      <td>NaN</td>
      <td>50</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>wedding ceremony</td>
    </tr>
    <tr>
      <th>21502</th>
      <td>1</td>
      <td>NaN</td>
      <td>42</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>building a real estate</td>
    </tr>
    <tr>
      <th>21510</th>
      <td>2</td>
      <td>NaN</td>
      <td>28</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>NaN</td>
      <td>car purchase</td>
    </tr>
  </tbody>
</table>
<p>1105 rows × 12 columns</p>
</div>




```python
# Memeriksa pola lainnya dengan detail income type retiree
df.loc[(df['income_type']=='retiree') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>NaN</td>
      <td>65</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>M</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>to have a wedding</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>NaN</td>
      <td>63</td>
      <td>secondary education</td>
      <td>1</td>
      <td>unmarried</td>
      <td>4</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>building a real estate</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>NaN</td>
      <td>54</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>retiree</td>
      <td>1</td>
      <td>NaN</td>
      <td>to have a wedding</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0</td>
      <td>NaN</td>
      <td>52</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>purchase of the house for my family</td>
    </tr>
    <tr>
      <th>145</th>
      <td>0</td>
      <td>NaN</td>
      <td>62</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>building a property</td>
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
      <th>21311</th>
      <td>0</td>
      <td>NaN</td>
      <td>49</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>buying property for renting out</td>
    </tr>
    <tr>
      <th>21321</th>
      <td>0</td>
      <td>NaN</td>
      <td>56</td>
      <td>Secondary Education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>real estate transactions</td>
    </tr>
    <tr>
      <th>21414</th>
      <td>0</td>
      <td>NaN</td>
      <td>65</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>purchase of my own house</td>
    </tr>
    <tr>
      <th>21415</th>
      <td>0</td>
      <td>NaN</td>
      <td>54</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>housing transactions</td>
    </tr>
    <tr>
      <th>21423</th>
      <td>0</td>
      <td>NaN</td>
      <td>63</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>retiree</td>
      <td>0</td>
      <td>NaN</td>
      <td>purchase of a car</td>
    </tr>
  </tbody>
</table>
<p>413 rows × 12 columns</p>
</div>




```python
# Memeriksa pola lainnya dengan detail income type business
df.loc[(df['income_type']=='business') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>0</td>
      <td>NaN</td>
      <td>21</td>
      <td>secondary education</td>
      <td>1</td>
      <td>unmarried</td>
      <td>4</td>
      <td>M</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>transactions with commercial real estate</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1</td>
      <td>NaN</td>
      <td>34</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>having a wedding</td>
    </tr>
    <tr>
      <th>121</th>
      <td>0</td>
      <td>NaN</td>
      <td>29</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>car</td>
    </tr>
    <tr>
      <th>135</th>
      <td>0</td>
      <td>NaN</td>
      <td>27</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>housing</td>
    </tr>
    <tr>
      <th>174</th>
      <td>0</td>
      <td>NaN</td>
      <td>55</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>widow / widower</td>
      <td>2</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>to own a car</td>
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
      <th>21390</th>
      <td>20</td>
      <td>NaN</td>
      <td>53</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>buy residential real estate</td>
    </tr>
    <tr>
      <th>21391</th>
      <td>0</td>
      <td>NaN</td>
      <td>52</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>purchase of the house for my family</td>
    </tr>
    <tr>
      <th>21407</th>
      <td>1</td>
      <td>NaN</td>
      <td>36</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>building a real estate</td>
    </tr>
    <tr>
      <th>21489</th>
      <td>2</td>
      <td>NaN</td>
      <td>47</td>
      <td>Secondary Education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>purchase of a car</td>
    </tr>
    <tr>
      <th>21497</th>
      <td>0</td>
      <td>NaN</td>
      <td>48</td>
      <td>BACHELOR'S DEGREE</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>NaN</td>
      <td>building a property</td>
    </tr>
  </tbody>
</table>
<p>508 rows × 12 columns</p>
</div>




```python
# Memeriksa pola lainnya dengan detail income type civil servant
df.loc[(df['income_type']=='civil cervant') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
 # Memeriksa pola lainnya dengan detail income type unemployed
df.loc[(df['income_type']=='unemployed') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
 # Memeriksa pola lainnya dengan detail income type entrepreneur
df.loc[(df['income_type']=='entrepreneur') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5936</th>
      <td>0</td>
      <td>NaN</td>
      <td>58</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>entrepreneur</td>
      <td>0</td>
      <td>NaN</td>
      <td>buy residential real estate</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Memeriksa pola lainnya dengan detail income type  student
df.loc[(df['income_type']=='student') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
 # Memeriksa pola lainnya dengan detail income type  'paternity / maternity leave'
df.loc[(df['income_type']=='paternity / maternity leave') & (df['total_income'].isna())]
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



**Kesimpulan**

[Apakah Anda menemukan suatu pola? Bagaimana Anda mendapatkan kesimpulan ini?]

Terdapat pola nilai yang hilang pada kategori income type :
1. kategori employee dengan nilai yang hilang sebanyak : 1105 rows
2. kategori retiree dengan nilai yang hilang sebanyak : 413 rows 
3. kategori business dengan nilai yang hilang sebanyak : 508 rows
4. kategori enterprenuer dengan nilai yang hilang sebanyak : 1 rows

[Jelaskan bagaimana Anda akan mengatasi nilai-nilai yang hilang. Mempertimbangkan kategori yang nilainya tidak ada.]

untuk mengatasi missing value pada kategori yang teridentifikasi memiliki nilai yang hilang sebaiknya kita kita memperbaiki data secara keseluruhan lalu mengisi nilai yang hilang dengan menghitung rata-rata atau median pada column yang nilainya hilang  berdasarkan umur yang dikelompokkan dan total penghasilan yang dikelompokkan.  

[Buatlah perencanaan secara singkat langkah Anda selanjutnya untuk mengubah data. Anda mungkin perlu mengatasi berbagai jenis masalah: duplikat, register yang berbeda, data lama yang salah, dan nilai yang hilang.]

Perencanaan secara singkat pada transformasi data dimana kita akan memeriksa value setiap column yang meliputi :
1.  Data yang terduplikasi
2.  Register yang berbeda
3.  Data lama yang salah / tidak wajar 
4.  mengisi nilai yang hilang

## Transformasi data

[Mari kita perhatikan setiap kolom untuk melihat masalah apa yang mungkin kita miliki di dalamnya.]

[Mulailah dengan menghapus duplikat dan memperbaiki informasi pendidikan jika diperlukan.]


```python
# Mari kita lihat semua nilai di kolom pendidikan untuk memeriksa ejaan apa yang perlu diperbaiki
sorted(df['education'].unique())

```




    ["BACHELOR'S DEGREE",
     "Bachelor's Degree",
     'GRADUATE DEGREE',
     'Graduate Degree',
     'PRIMARY EDUCATION',
     'Primary Education',
     'SECONDARY EDUCATION',
     'SOME COLLEGE',
     'Secondary Education',
     'Some College',
     "bachelor's degree",
     'graduate degree',
     'primary education',
     'secondary education',
     'some college']



Terdapat data pada bebrapa value dengan register yang salah sehingga harus di terapkan metode yang dapat menyamakan value nya


```python
# Memeriksa data yang terduplikasi
df['education'].duplicated()
```




    0        False
    1        False
    2        False
    3         True
    4         True
             ...  
    21520     True
    21521     True
    21522     True
    21523     True
    21524     True
    Name: education, Length: 21525, dtype: bool




```python
# Perbaiki register jika diperlukan
df['education'] = df['education'].str.lower()

```


```python
# Memeriksa perbaikan terhadap duplikasi data
sorted(df['education'].unique())
```




    ["bachelor's degree",
     'graduate degree',
     'primary education',
     'secondary education',
     'some college']



Setelah penggunaan str.lower() value dengan register yang salah telah disamakan.


```python
# Memeriksa data yang terduplikasi 
df['education'].duplicated()
```




    0        False
    1        False
    2         True
    3         True
    4         True
             ...  
    21520     True
    21521     True
    21522     True
    21523     True
    21524     True
    Name: education, Length: 21525, dtype: bool



[Periksa data kolom `children`]


```python
# Mari kita lihat distribusi nilai pada kolom `children`
sorted(df['children'].unique())
```




    [-1, 0, 1, 2, 3, 4, 5, 20]



[Apakah terdapat hal-hal aneh di kolom? Jika jawabannya iya, seberapa tinggi persentase data yang bermasalah? Bagaimana mereka bisa terjadi? Buat keputusan tentang apa yang akan Anda lakukan dengan data ini dan jelaskan alasannya.]

**terdapat nilai numerik dengan value yang kurang tepat yaitu -1, dimana jika seseorang tidak mempunyai anak dengan nilai 0 dan nilai 20 menjadi 2**


```python
# [perbaiki data berdasarkan keputusan Anda]
df['children'] = df['children'].replace(-1,0)
df['children'] = df['children'].replace(20,2)
```


```python
# Periksa kembali kolom `children` untuk memastikan semua telah diperbaiki

sorted(df['children'].unique())
```




    [0, 1, 2, 3, 4, 5]



[Periksa data dalam kolom the `days_employed`. Pertama-tama pikirkan tentang masalah apa yang mungkin ada dan apa yang mungkin ingin Anda periksa dan bagaimana Anda akan melakukannya.]


```python
# Temukan data yang bermasalah di `days_employed`, jika terdapat masalah, dan hitung persentasenya
# Menghitung lama nasabah bekerja dengan nilai negatid dibawah 0 tahun
day_min = df['days_employed'].loc[(df['days_employed'] < 0)].count() 
print(f'jumlah lama bekerja kurang dari 0 hari adalah : {day_min} row')
```

    jumlah lama bekerja kurang dari 0 hari adalah : 15906 row



```python
# Menghitung lama nasabah yang melebihi batas usia nasabah
day_max = df['days_employed'].loc[(df['days_employed'] > 14610)].count()
print(f'jumlah lama bekerja melebihi dari 14610 hari adalah : {day_max} row')
```

    jumlah lama bekerja melebihi dari 14610 hari adalah : 3445 row



```python
# Menampilkan persentase error dari column days employed
perc_of_error = (day_min + day_max) / len(df) * 100
print(f'persentasi dari value yang error adalah : {perc_of_error}%')
```

    persentasi dari value yang error adalah : 89.90011614401858%


[Jika jumlah data yang bermasalah tinggi, hal tersebut mungkin dikarenakan beberapa masalah teknis. Kami mungkin ingin mengusulkan alasan paling jelas mengapa hal tersebut dapat terjadi dan bagaimanakah kemungkinan data yang benar, karena kita tidak dapat menghapus baris yang bermasalah ini.]

***Terdapat data yang tidak wajar dengan nilai negatif sebesar 15906 row dan lama nasabah bekerja diatas 40 tahun sebesar 3455 row***

***Data yang bermasalah pada days_employed berdasarakan persentasenya adalah ; batas wajar bekerja dari rata-rata umur nasabah dikali dengan 365 hari, lalu ditambahkan dengan persentase dari nilai yang memiliki nilai negatif dibagi dengan panjang row dikalikan 100. maka hasilnya adalah 89%.***



```python
# Atasi nilai yang bermasalah, jika ada
#langkah pertama merubah nilai yang negativ menjadi positif
df['days_employed'] = abs(df['days_employed'])
```


```python
#mengganti nilai berapa lama nasabah telah bekerja dalam hari dengan maksimal 40 tahun rata-rata nasabah bekerja
df.loc[df['days_employed'] > 14610, 'days_employed'] = 14610
```


```python
# Periksa hasilnya - pastikan telah diperbaiki
print('menampilkan jumlah nasabah yang bekerja diatas 40 Tahun')
print(df['days_employed'].loc[df['days_employed'] > 15585].count())
```

    menampilkan jumlah nasabah yang bekerja diatas 40 Tahun
    0



```python
# Memeriksa distribusi pada column days employed
print('Menampilkan distribusi dari column days_employed :')
print(df['days_employed'].describe())
```

    Menampilkan distribusi dari column days_employed :
    count    19351.000000
    mean      4534.078277
    std       5131.000987
    min         24.141633
    25%        927.009265
    50%       2194.220567
    75%       5537.882441
    max      14610.000000
    Name: days_employed, dtype: float64


[Sekarang mari kita melihat usia klien dan apakah terdapat masalah di sana. Sekali lagi, pikirkan tentang data apakah yang dapat menjadi suatu kejanggalan pada kolom ini, yaitu berapa usia seseorang.]


```python
# Periksa `dob_years` untuk nilai yang mencurigakan dan hitung persentasenya
print('memeriksa nilai yang unik pada column dob_years:')
print(df['dob_years'].unique())
```

    memeriksa nilai yang unik pada column dob_years:
    [42 36 33 32 53 27 43 50 35 41 40 65 54 56 26 48 24 21 57 67 28 63 62 47
     34 68 25 31 30 20 49 37 45 61 64 44 52 46 23 38 39 51  0 59 29 60 55 58
     71 22 73 66 69 19 72 70 74 75]



```python
# Memeriksa distribusi nilai pada column dob years
print('memeriksa distribusi nilai column dob_years:')
print(df['dob_years'].describe())
```

    memeriksa distribusi nilai column dob_years:
    count    21525.000000
    mean        43.293380
    std         12.574584
    min          0.000000
    25%         33.000000
    50%         42.000000
    75%         53.000000
    max         75.000000
    Name: dob_years, dtype: float64



```python
# Menghitung nilai umur nasabah yang teridentifikasi 0 tahun
df_dy_eror = df['dob_years'].loc[df['dob_years'] == 0].count()
print(df_dy_eror)
```

    101



```python
# Menghitung Persentase nilai error pada umur nasabah
persentase_dy_eror = (df_dy_eror / len(df)) * 100
persentase_dy_eror
```




    0.4692218350754936



[Putuskan apa yang akan Anda lakukan dengan nilai yang bermasalah dan jelaskan alasannya.]

**Terdapat nilai yang unik dan umur nasabah dimana nilainya adalah  = 0, mari kita hapus nilai yang terduplikasi dan ganti nilai yang = 0 dengan rata-rata umur nasabah.**


```python
# Atasi masalah pada kolom `dob_years`, jika terdapat masalah
print('mengganti nilai dob years dari 0 menjadi 43 tahun')
df.loc[df['dob_years'] == 0, 'dob_years'] = 43 #rata-rata umur nasabah

```

    mengganti nilai dob years dari 0 menjadi 43 tahun


Alasan terkait penggunaan rata-rata yakni 43 dibanding nilainya adalah pendekatan secara statistika dimana dalam menangani nilai yang hilang kita dapat menentukan nilai yang hilang dengan mean/median/modus. disini saya memilih nilai mean atau rata-rata umur nasabah 43 tahun dikarenakan rata-rata nasabah memiliki umur 43 tahun. 


```python
# Periksa hasilnya - pastikan telah diperbaiki
print('Memeriksa hasil dari nilai yang telah diperbaiki :')
print(df.loc[df['dob_years'] == 0])
```

    Memeriksa hasil dari nilai yang telah diperbaiki :
    Empty DataFrame
    Columns: [children, days_employed, dob_years, education, education_id, family_status, family_status_id, gender, income_type, debt, total_income, purpose]
    Index: []


[Sekarang saatnya memeriksa kolom `family_status`. Lihat nilai seperti apakah yang terdapat di kolom dan masalah apa yang mungkin perlu Anda atasi.]


```python
# Mari kita lihat nilai untuk kolom
print('memeriksa nilai yang unik :')
print(df['family_status'].unique())

```

    memeriksa nilai yang unik :
    ['married' 'civil partnership' 'widow / widower' 'divorced' 'unmarried']



```python
# Atasi nilai yang bermasalah di `family_status`, jika ada
#tidak ada value yang bermasalah di column famil_status

```


```python
# Periksa hasilnya - pastikan nilai telah diperbaiki

```

[Sekarang saatnya memeriksa kolom `gender`. Lihat nilai seperti apakah yang terdapat di kolom dan masalah apa yang mungkin perlu Anda atasi.]


```python
# Mari kita melihat nilainya di kolom
print(df['gender'].unique())
```

    ['F' 'M' 'XNA']



```python
# Menghitung nilai gender yang XNA
print(df['gender'].loc[df['gender'] == 'XNA'].count())
```

    1



```python
# Atasi nilai-nilai yang bermasalah, jika ada
df.loc[df['gender']== 'XNA', 'gender']='F'
```


```python
# Periksa hasilnya - pastikan telah diperbaiki
print(df['gender'].loc[df['gender'] == 'XNA'].count())
```

    0


[Sekarang saatnya memeriksa kolom `income_type`. Lihat nilai seperti apakah yang terdapat di kolom dan masalah apa yang mungkin perlu Anda atasi.]


```python
# Mari kita lihat nilai dalam kolom income type
print(df['income_type'].unique())
```

    ['employee' 'retiree' 'business' 'civil servant' 'unemployed'
     'entrepreneur' 'student' 'paternity / maternity leave']



```python
# Atasi nilai yang bermasalah, jika ada
#tidak ada value yang bermasalah di column income_type
```


```python
# Periksa hasilnya - pastikan telah diperbaiki


```

[Sekarang saatnya melihat apakah terdapat duplikasi di dalam data kita. Jika kita menemukannya, Anda harus memutuskan apa yang akan Anda lakukan dengan duplikat tersebut dan menjelaskan alasannya.]


```python
# Memeriksa duplikat 
print(df.duplicated().sum())
```

    71



```python
# Atasi duplikat, jika ada

df = df.drop_duplicates().reset_index(drop=True) 
```


```python
# Terakhir periksa apakah kita memiliki duplikat
df.duplicated().sum()
```




    0




```python
# Periksa ukuran dataset yang sekarang Anda miliki setelah manipulasi pertama yang Anda lakukan
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21454 entries, 0 to 21453
    Data columns (total 12 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   children          21454 non-null  int64  
     1   days_employed     19351 non-null  float64
     2   dob_years         21454 non-null  int64  
     3   education         21454 non-null  object 
     4   education_id      21454 non-null  int64  
     5   family_status     21454 non-null  object 
     6   family_status_id  21454 non-null  int64  
     7   gender            21454 non-null  object 
     8   income_type       21454 non-null  object 
     9   debt              21454 non-null  int64  
     10  total_income      19351 non-null  float64
     11  purpose           21454 non-null  object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 2.0+ MB



```python
# Menampilkan Persentase dari dataset awal dengan dataset yang telah di manipulasi
df_percentage_after = 21525 - len(df)
print(f'Menampilkan ukuran dataset yang telah di manipulasi dengan jumlah : {df_percentage_after} row')
```

    Menampilkan ukuran dataset yang telah di manipulasi dengan jumlah : 71 row



```python
# Persentase akhir dari perbandingan dataset awal dengan dataset yang sudah dimanipulasi
print(f'Persentase akhir dari perbandingan dataset awal dengan dataset yang sudah dimanipulasi : {df_percentage_after / len(df) *100}%')

```

    Persentase akhir dari perbandingan dataset awal dengan dataset yang sudah dimanipulasi : 0.3309406171343339%



```python
# Menampilakan distribusi jumlah row dari datasets yang telah di manipulasi 
df.shape
```




    (21454, 12)



[Jelaskan dataset baru Anda: jelaskan secara singkat apa perubahannya dan berapa persentase perubahannya, jika ada.]


**Terdapat Perubahan jumlah dataset awal yang berjumlah 21525 setelah dimanipulasi menjadi 21454 dengan persentase perbedaan dataset sebesar 0.3%**


# Bekerja dengan nilai yang hilang

[Untuk mempercepat pekerjaan dengan beberapa data, Anda mungkin ingin menggunakan dictionary untuk beberapa nilai, di mana tersedia ID. Jelaskan mengapa dan dictionary apakah yang akan Anda gunakan.]

**Untuk penggunaan dictionary untuk beberapa nilai kita dapat mengelompokkan umur nasabah untuk penyedian ID agar mempermudah pengisian nilai yang hilang.**


```python
# Temukan dictionary
sorted(df['dob_years'].unique())
```




    [19,
     20,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28,
     29,
     30,
     31,
     32,
     33,
     34,
     35,
     36,
     37,
     38,
     39,
     40,
     41,
     42,
     43,
     44,
     45,
     46,
     47,
     48,
     49,
     50,
     51,
     52,
     53,
     54,
     55,
     56,
     57,
     58,
     59,
     60,
     61,
     62,
     63,
     64,
     65,
     66,
     67,
     68,
     69,
     70,
     71,
     72,
     73,
     74,
     75]



### Memperbaiki nilai yang hilang di `total_income`

[Jelaskan secara singkat kolom manakah yang memiliki nilai yang hilang yang perlu Anda tangani. Jelaskan bagaimana Anda akan memperbaikinya.]

**Column dengan nilai yang hilang ada pada column days_employed & total_income, saya akan mengisi nilai yang hilang dengan mengkategorikan usia nasabah lalu mencari nilai rata/median untuk mengisi nilai yang hilang pada pendapatan total & lama nasabah bekerja**

[Mulailah dengan mengatasi nilai pendapatan total yang hilang. Membuat kategori usia untuk klien. Membuat kolom baru dengan kategori usia. Strategi ini dapat dibantu dengan menghitung nilai pendapatan total.]



```python
# Mari menulis fungsi untuk menghitung kategori usia
age = df['dob_years']

def age_group(age):
    if age < 15 :
        return '15 Tahun Kebawah'
    elif age >=16 and age <=24:
        return '16 Tahun - 24 Tahun'
    elif age >= 25 and age <=34:
        return '25 Tahun - 34 Tahun'
    elif age >= 35 and age <=44:
        return '35 Tahun - 44 Tahun'
    elif age >= 45 and age <=54:
        return '35 Tahun - 44 Tahun'
    elif age >= 55 and age <=64:
        return '55 Tahun - 64 Tahun'
    else :
        return '65 Tahun Keatas'

```


```python
# Lakukan pengujian apakah fungsi bekerja atau tidak
age_group (30)
```




    '25 Tahun - 34 Tahun'




```python
# Membuat kolom baru berdasarkan fungsi
df['age_group'] = df['dob_years'].apply(age_group)
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8437.673028</td>
      <td>42</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>40620.102</td>
      <td>purchase of the house</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4024.803754</td>
      <td>36</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>17932.802</td>
      <td>car purchase</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5623.422610</td>
      <td>33</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>23341.752</td>
      <td>purchase of the house</td>
      <td>25 Tahun - 34 Tahun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4124.747207</td>
      <td>32</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>42820.568</td>
      <td>supplementary education</td>
      <td>25 Tahun - 34 Tahun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>14610.000000</td>
      <td>53</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>25378.572</td>
      <td>to have a wedding</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Memeriksa bagaimana nilai di dalam kolom baru
df['age_group'].value_counts()
```




    35 Tahun - 44 Tahun    10708
    25 Tahun - 34 Tahun     5092
    55 Tahun - 64 Tahun     3884
    65 Tahun Keatas          895
    16 Tahun - 24 Tahun      875
    Name: age_group, dtype: int64



[Pikirkan tentang faktor-faktor yang biasanya bergantung pada pendapatan. Akhirnya, Anda akan mengetahui apakah Anda harus menggunakan nilai rata-rata atau median untuk mengganti nilai yang hilang. Untuk membuat keputusan ini, Anda mungkin ingin melihat identifikasi distribusi faktor yang memengaruhi pendapatan seseorang.]

[Buat tabel yang hanya memiliki data tanpa nilai yang hilang. Data ini akan digunakan untuk memperbaiki nilai yang hilang.]


```python
# Membuat tabel tanpa nilai yang hilang dan menampilkan beberapa barisnya untuk memastikan semuanya berjalan dengan baik
#df_dropna = df.loc[~(df['days_employed'].()) & ~(df['total_income'].isna())]
df_dropna = df.dropna()
df_dropna
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8437.673028</td>
      <td>42</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>40620.102</td>
      <td>purchase of the house</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4024.803754</td>
      <td>36</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>17932.802</td>
      <td>car purchase</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5623.422610</td>
      <td>33</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>23341.752</td>
      <td>purchase of the house</td>
      <td>25 Tahun - 34 Tahun</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4124.747207</td>
      <td>32</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>42820.568</td>
      <td>supplementary education</td>
      <td>25 Tahun - 34 Tahun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>14610.000000</td>
      <td>53</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>25378.572</td>
      <td>to have a wedding</td>
      <td>35 Tahun - 44 Tahun</td>
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
      <th>21449</th>
      <td>1</td>
      <td>4529.316663</td>
      <td>43</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>business</td>
      <td>0</td>
      <td>35966.698</td>
      <td>housing transactions</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>21450</th>
      <td>0</td>
      <td>14610.000000</td>
      <td>67</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>24959.969</td>
      <td>purchase of a car</td>
      <td>65 Tahun Keatas</td>
    </tr>
    <tr>
      <th>21451</th>
      <td>1</td>
      <td>2113.346888</td>
      <td>38</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>M</td>
      <td>employee</td>
      <td>1</td>
      <td>14347.610</td>
      <td>property</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>21452</th>
      <td>3</td>
      <td>3112.481705</td>
      <td>38</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>1</td>
      <td>39054.888</td>
      <td>buying my own car</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
    <tr>
      <th>21453</th>
      <td>2</td>
      <td>1984.507589</td>
      <td>40</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>13127.587</td>
      <td>to buy a car</td>
      <td>35 Tahun - 44 Tahun</td>
    </tr>
  </tbody>
</table>
<p>19351 rows × 13 columns</p>
</div>




```python
# Perhatikan nilai rata-rata untuk pendapatan berdasarkan faktor yang telah Anda identifikasi
print('nilai mean dari pendapatan adalah:')
df_dropna.pivot_table(index='age_group', values='total_income', aggfunc='mean')
```

    nilai mean dari pendapatan adalah:





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
      <th>total_income</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16 Tahun - 24 Tahun</th>
      <td>22703.351103</td>
    </tr>
    <tr>
      <th>25 Tahun - 34 Tahun</th>
      <td>27337.934929</td>
    </tr>
    <tr>
      <th>35 Tahun - 44 Tahun</th>
      <td>28087.460887</td>
    </tr>
    <tr>
      <th>55 Tahun - 64 Tahun</th>
      <td>24601.730826</td>
    </tr>
    <tr>
      <th>65 Tahun Keatas</th>
      <td>21542.650450</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Perhatikan nilai median untuk pendapatan berdasarkan faktor yang telah Anda identifikasi
print('nilai median dari pendapatan adalah:')
df_dropna.pivot_table(index='age_group', values='total_income', aggfunc='median')
```

    nilai median dari pendapatan adalah:





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
      <th>total_income</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16 Tahun - 24 Tahun</th>
      <td>20572.209</td>
    </tr>
    <tr>
      <th>25 Tahun - 34 Tahun</th>
      <td>23990.901</td>
    </tr>
    <tr>
      <th>35 Tahun - 44 Tahun</th>
      <td>24278.732</td>
    </tr>
    <tr>
      <th>55 Tahun - 64 Tahun</th>
      <td>21339.562</td>
    </tr>
    <tr>
      <th>65 Tahun Keatas</th>
      <td>18471.391</td>
    </tr>
  </tbody>
</table>
</div>



[Ulangi perbandingan tersebut untuk beberapa faktor. Pastikan Anda mempertimbangkan berbagai aspek dan menjelaskan proses pada saat Anda berpikir.]



[Buat keputusan tentang karakteristik yang paling menentukan pendapatan dan apakah Anda akan menggunakan median atau mean. Jelaskan mengapa Anda membuat keputusan ini]

**Dilihat dari tabel disttribusi 'total_income' , saya menggunakan median untuk mengisi nilai yang hilang. karena terdapat nilai pada median pada age_group nilai rentang distribusinya tidak terlalu jauh**


```python
# Mari tulis fungsi yang menghitung rata-rata atau median (tergantung keputusan Anda) berdasarkan parameter yang Anda identifikasi

df['total_income'] = df.groupby('age_group')['total_income'].transform(lambda x: x.fillna(x.median()))

```


```python
# Memeriksa bagaimana nilai di dalam kolom baru
df['total_income'].isna().sum()
```




    0



nilai yang hilang pada column total income sekarang sudah terisi berdasarkan parameter yang telah diindetifikasi


```python
# Terapkan fungsi ke setiap baris
#fungsi sudah diterapkan pada column sebelumnya
```


```python
# Periksa apakah kita mendapatkan kesalahan
#fungsi sudah di periksa pada column sebelumnya
```

[Jika Anda menemukan kesalahan dalam menyiapkan nilai data yang hilang, artinya mungkin terdapat sesuatu yang khusus terkait dengan data untuk kategori tersebut. Mari pikirkan - Anda mungkin ingin memperbaiki beberapa hal secara manual, jika terdapat cukup data untuk menemukan median/rata-rata.]



```python
# Mengganti nilai yang hilang jika terdapat kesalahan
#Nilai yang hilang sudah digantikan pada column diatas
```

[Ketika Anda berpikir Anda telah selesai dengan `total_income`, periksa apakah jumlah total nilai di kolom ini sesuai dengan jumlah nilai di kolom lain.]


```python
# Memeriksa jumlah entri di kolom
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21454 entries, 0 to 21453
    Data columns (total 13 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   children          21454 non-null  int64  
     1   days_employed     19351 non-null  float64
     2   dob_years         21454 non-null  int64  
     3   education         21454 non-null  object 
     4   education_id      21454 non-null  int64  
     5   family_status     21454 non-null  object 
     6   family_status_id  21454 non-null  int64  
     7   gender            21454 non-null  object 
     8   income_type       21454 non-null  object 
     9   debt              21454 non-null  int64  
     10  total_income      21454 non-null  float64
     11  purpose           21454 non-null  object 
     12  age_group         21454 non-null  object 
    dtypes: float64(2), int64(5), object(6)
    memory usage: 2.1+ MB


###  Memperbaiki nilai di `days_employed`

[Pikirkan tentang parameter yang dapat membantu Anda memperbaiki nilai yang hilang di kolom ini. Akhirnya, Anda akan mengetahui apakah Anda harus menggunakan nilai rata-rata atau median untuk mengganti nilai yang hilang. Anda mungkin akan melakukan penelitian yang sama dengan yang Anda lakukan saat memperbaiki data di kolom sebelumnya.]


```python
# Distribusi median dari `days_employed` berdasarkan parameter yang Anda identifikasi
print('dirstribusi dari nilai median days_employed berdasarkan age_group')
df.pivot_table(index='age_group', values='days_employed', aggfunc='median')
```

    dirstribusi dari nilai median days_employed berdasarkan age_group





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
      <th>days_employed</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16 Tahun - 24 Tahun</th>
      <td>744.542130</td>
    </tr>
    <tr>
      <th>25 Tahun - 34 Tahun</th>
      <td>1292.221018</td>
    </tr>
    <tr>
      <th>35 Tahun - 44 Tahun</th>
      <td>2180.412650</td>
    </tr>
    <tr>
      <th>55 Tahun - 64 Tahun</th>
      <td>14610.000000</td>
    </tr>
    <tr>
      <th>65 Tahun Keatas</th>
      <td>14610.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Distribusi rata-rata dari `days_employed` berdasarkan parameter yang Anda identifikasi
print('dirstribusi dari nilai mean days_employed berdasarkan age_group')
df.pivot_table(index='age_group', values='days_employed', aggfunc='mean')
```

    dirstribusi dari nilai mean days_employed berdasarkan age_group





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
      <th>days_employed</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16 Tahun - 24 Tahun</th>
      <td>871.669342</td>
    </tr>
    <tr>
      <th>25 Tahun - 34 Tahun</th>
      <td>1629.238372</td>
    </tr>
    <tr>
      <th>35 Tahun - 44 Tahun</th>
      <td>3464.105828</td>
    </tr>
    <tr>
      <th>55 Tahun - 64 Tahun</th>
      <td>10158.577181</td>
    </tr>
    <tr>
      <th>65 Tahun Keatas</th>
      <td>13092.784889</td>
    </tr>
  </tbody>
</table>
</div>



[Tentukan apa yang akan Anda gunakan: rata-rata atau median. Jelaskan mengapa.]

**Dilihat dari distribusi days_employed, saya menggunakan median untuk mengisi nilai yang hilang, dikarenakan terdapat value yang `outlier` dimana value dari days_employed terdapat rentang nilai yang terlalu rendah.**


```python
# Mari tulis fungsi yang menghitung rata-rata atau median (tergantung keputusan Anda) berdasarkan parameter yang Anda identifikasi

df['days_employed'] = df.groupby('age_group')['days_employed'].transform(lambda x: x.fillna(x.median()))
   
```


```python
# Periksa bahwa fungsi bekerja
df['days_employed'].isna().sum()
```




    0




```python
# Terapkan fungsi ke income_type
#fungsi sudah diterapkan pada column sebelumnya
```


```python
# Periksa bahwa fungsi bekerja
#Fungsi sudah diperiksa di column atas
```


```python
# Mengganti nilai yang hilang jika ada terdapat kesalahan
#Nilai yang hilang sudah digantikan pada column diatas


```

[Ketika Anda berpikir bahwa Anda telah selesai dengan `total_income`, periksa apakah jumlah total nilai di kolom ini sesuai dengan jumlah nilai di kolom lain.]


```python
# Periksa entri di semua kolom - pastikan kita memperbaiki semua nilai yang hilang
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21454 entries, 0 to 21453
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   children          21454 non-null  int64  
     1   days_employed     21454 non-null  float64
     2   dob_years         21454 non-null  int64  
     3   education         21454 non-null  object 
     4   education_id      21454 non-null  int64  
     5   family_status     21454 non-null  object 
     6   family_status_id  21454 non-null  int64  
     7   gender            21454 non-null  object 
     8   income_type       21454 non-null  object 
     9   debt              21454 non-null  int64  
     10  total_income      21454 non-null  float64
     11  purpose           21454 non-null  object 
     12  age_group         21454 non-null  object 
     13  level_income      21454 non-null  object 
     14  gender_meaning    21454 non-null  object 
     15  debt_meaning      21454 non-null  object 
    dtypes: float64(2), int64(5), object(9)
    memory usage: 2.6+ MB


## 4. Pengkategorian Data

[Untuk menjawab pertanyaan dan menguji hipotesis, Anda akan bekerja dengan data yang telah dikategorikan. Lihatlah pertanyaan-pertanyaan yang diajukan kepada Anda dan yang harus Anda jawab. Pikirkan tentang data mana yang perlu dikategorikan untuk menjawab pertanyaan-pertanyaan ini. Di bawah ini Anda akan menemukan template di mana Anda dapat bekerja dengan cara Anda sendiri saat mengkategorikan data. Proses pertama adalah menutup data teks; yang kedua mengatasi data numerik yang perlu dikategorikan. Anda dapat menggunakan keduanya atau tidak sama sekali dari petunjuk yang disarankan - terserah Anda.]

[Terlepas dari keputusan Anda untuk mengatasi pengkategorian, pastikan secara jelas Anda memberikan penjelasan tentang mengapa Anda membuat keputusan tersebut. Ingat: ini merupakan pekerjaan Anda dan semua di dalamnya adalah keputusan Anda.]

Langkah saya dalam pengkategorian data pada bab ini untuk menjawab pertanyaan dari hipotesis adalah  :
1. Menampilkan data pengkategorian untuk menjawab hipotesis
2. Melihat nilai unik pada setiap kategori yang telah dipilih untuk di identifikasi berdasarkan nilai uniknya yakni total income, gender, debt.
3. Mengkategorikan value total income menjadi tingkat pendapatan.
4. Mengkategorikan value gender menjadi gender meaning untuk di klasifisikan menjadi Female & Male.
5. Mengkategorikan value debt 0 untuk berhasil melunasi pinjaman dan 1 untuk gala melunasi pinjaman. 
6. Mengkategorikan value purpose agar lebih rapi

penjelasan mengenai keputusan yang diambil dalam pengkategorian data menggunakan data categorical akan lebih mudah dalam menjawab pertanyaan dari hipotesis. 
 


```python
# Tampilkan nilai data yang Anda pilih untuk pengkategorian
category_column = df[['total_income', 'gender', 'debt','purpose']]
category_column
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
      <th>total_income</th>
      <th>gender</th>
      <th>debt</th>
      <th>purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40620.102</td>
      <td>F</td>
      <td>0</td>
      <td>purchase of the house</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17932.802</td>
      <td>F</td>
      <td>0</td>
      <td>car purchase</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23341.752</td>
      <td>M</td>
      <td>0</td>
      <td>purchase of the house</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42820.568</td>
      <td>M</td>
      <td>0</td>
      <td>supplementary education</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25378.572</td>
      <td>F</td>
      <td>0</td>
      <td>to have a wedding</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21449</th>
      <td>35966.698</td>
      <td>F</td>
      <td>0</td>
      <td>housing transactions</td>
    </tr>
    <tr>
      <th>21450</th>
      <td>24959.969</td>
      <td>F</td>
      <td>0</td>
      <td>purchase of a car</td>
    </tr>
    <tr>
      <th>21451</th>
      <td>14347.610</td>
      <td>M</td>
      <td>1</td>
      <td>property</td>
    </tr>
    <tr>
      <th>21452</th>
      <td>39054.888</td>
      <td>M</td>
      <td>1</td>
      <td>buying my own car</td>
    </tr>
    <tr>
      <th>21453</th>
      <td>13127.587</td>
      <td>F</td>
      <td>0</td>
      <td>to buy a car</td>
    </tr>
  </tbody>
</table>
<p>21454 rows × 4 columns</p>
</div>



[Let's check unique values]



```python
category_column = df[['total_income', 'gender', 'debt','purpose']]
for loop in category_column:
    print(loop)
    print(df[loop].unique())
    print()
```

    total_income
    [40620.102 17932.802 23341.752 ... 14347.61  39054.888 13127.587]
    
    gender
    ['F' 'M']
    
    debt
    [0 1]
    
    purpose
    ['Buy Property Purpose' 'Buy Car Purpose' 'Education Purpose'
     'Wedding Purpose' 'Building Property Purpose']
    


[Kelompok utama apakah yang dapat Anda identifikasi berdasarkan nilai uniknya?

**kelompok utama yang dapat di identifikasi berdasarkan nilai uniknya adalah total income, gender, debt, purpose**

[Berdasarkan topik ini, kita ingin mengkategorikan data kita.]




```python
# Membuat fungsi untuk mengkategorikan total pendapatan menjadi tingkat pendapatan
tingkat_pendapatan = df['total_income'] 

def level_of_income(tingkat_pendapatan):
    if tingkat_pendapatan >= 0 and tingkat_pendapatan <= 1035:
        return 'Low Income'
    elif tingkat_pendapatan >= 1036 and tingkat_pendapatan <= 4045:
        return 'Lower Middle Income'
    elif tingkat_pendapatan >= 4046 and tingkat_pendapatan <=12535: 
        return 'Upper Middle Income'
    elif tingkat_pendapatan >= 12536:
        return 'High Income'
    else :
        return 'unknown'
```


```python
# Menerapkan fungsi untuk dibuat column baru ke dataset untuk pengkategorian tingkat pendapatan berdasarkan total income
df['level_income'] = df['total_income'].apply(level_of_income)
```


```python
# Menghitung nilai pada column level_income
df['level_income'].count()
```




    21454




```python
# Membuat fungsi untuk mengkategorikan data berdasarkan gender
def gender_meaning(value):
    if value == 'F':
        return 'Female'
    else:
        return 'Male'
        
```


```python
# Buat kolom dengan kategori dan hitung nilainya
df['gender_meaning'] = df['gender'].apply(gender_meaning)
```


```python
# Menghitung nilai pada column gender meaning 
df['gender_meaning'].count()
```




    21454




```python
# Membuat fungsi untuk mengkategorikan data berdasarkan debt
def debt_meaning(value):
    if value == 1:
        return 'Gagal Melunasi Pinjaman'
    else:
        return 'Berhasil Melunasi Pinjaman'
```


```python
# Buat kolom dengan kategori debt mening dan hitung nilainya
df['debt_meaning'] = df['debt'].apply(debt_meaning)
```


```python
# Menghitung nilai pada column debt meaning 
df['debt_meaning'].count()
```




    21454




```python
# Memanipulasi Tujuan Kredit agar terlihat lebih rapi
df.loc[df['purpose'].isin(['building a property', 'building a real estate', 'construction of own property', 'housing renovation' ]), 'purpose'] = 'Building Property Purpose' 
df.loc[df['purpose'].isin(['property','buy commercial real estate', 'buy real estate', 'buy residential real estate', 'buying property for renting out', 'housing', 'housing transactions', 'purchase of my own house', 'purchase of the house', 'real estate transactions', 'transactions with commercial real estate','transactions with my real estate', 'purchase of the house for my family', ]), 'purpose'] = 'Buy Property Purpose' 
df.loc[df['purpose'].isin(['buying a second-hand car', 'buying my own car','car','car purchase','cars','purchase of a car','second-hand car purchase','second-hand car purchase','to buy a car','to own a car']), 'purpose'] = 'Buy Car Purpose' 
df.loc[df['purpose'].isin(['education', 'getting an education','getting higher education','going to university', 'supplementary education','to become educated', 'profile education', 'to get a supplementary education','university education',]), 'purpose'] = 'Education Purpose'
df.loc[df['purpose'].isin(['having a wedding','to have a wedding', 'wedding ceremony']), 'purpose'] = 'Wedding Purpose'

# Menampilkan hasil manipulasi purpose
sorted(df['purpose'].unique())
```




    ['Building Property Purpose',
     'Buy Car Purpose',
     'Buy Property Purpose',
     'Education Purpose',
     'Wedding Purpose']




```python
# Melihat dataset keseluruhan setelah di kategorikan
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
      <th>children</th>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>education</th>
      <th>education_id</th>
      <th>family_status</th>
      <th>family_status_id</th>
      <th>gender</th>
      <th>income_type</th>
      <th>debt</th>
      <th>total_income</th>
      <th>purpose</th>
      <th>age_group</th>
      <th>level_income</th>
      <th>gender_meaning</th>
      <th>debt_meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8437.673028</td>
      <td>42</td>
      <td>bachelor's degree</td>
      <td>0</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>40620.102</td>
      <td>Buy Property Purpose</td>
      <td>35 Tahun - 44 Tahun</td>
      <td>High Income</td>
      <td>Female</td>
      <td>Berhasil Melunasi Pinjaman</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4024.803754</td>
      <td>36</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>F</td>
      <td>employee</td>
      <td>0</td>
      <td>17932.802</td>
      <td>Buy Car Purpose</td>
      <td>35 Tahun - 44 Tahun</td>
      <td>High Income</td>
      <td>Female</td>
      <td>Berhasil Melunasi Pinjaman</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>5623.422610</td>
      <td>33</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>23341.752</td>
      <td>Buy Property Purpose</td>
      <td>25 Tahun - 34 Tahun</td>
      <td>High Income</td>
      <td>Male</td>
      <td>Berhasil Melunasi Pinjaman</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4124.747207</td>
      <td>32</td>
      <td>secondary education</td>
      <td>1</td>
      <td>married</td>
      <td>0</td>
      <td>M</td>
      <td>employee</td>
      <td>0</td>
      <td>42820.568</td>
      <td>Education Purpose</td>
      <td>25 Tahun - 34 Tahun</td>
      <td>High Income</td>
      <td>Male</td>
      <td>Berhasil Melunasi Pinjaman</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>14610.000000</td>
      <td>53</td>
      <td>secondary education</td>
      <td>1</td>
      <td>civil partnership</td>
      <td>1</td>
      <td>F</td>
      <td>retiree</td>
      <td>0</td>
      <td>25378.572</td>
      <td>Wedding Purpose</td>
      <td>35 Tahun - 44 Tahun</td>
      <td>High Income</td>
      <td>Female</td>
      <td>Berhasil Melunasi Pinjaman</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Memeriksa Hipotesis


**Apakah terdapat korelasi antara memiliki anak dengan membayar kembali tepat waktu?**


```python
# Periksa data anak dan membayar kembali dengan tepat waktu
# Menghitung tarif otomatis berdasarkan jumlah anak
df1 = df.pivot_table(index='children',columns='debt_meaning', values='debt', aggfunc='count') 
df1 = df1.reset_index()
df1['percentage'] = df1['Gagal Melunasi Pinjaman'] / (df1['Gagal Melunasi Pinjaman'] + df1['Berhasil Melunasi Pinjaman'])
df1
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
      <th>debt_meaning</th>
      <th>children</th>
      <th>Berhasil Melunasi Pinjaman</th>
      <th>Gagal Melunasi Pinjaman</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13074.0</td>
      <td>1064.0</td>
      <td>0.075258</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4364.0</td>
      <td>444.0</td>
      <td>0.092346</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1926.0</td>
      <td>202.0</td>
      <td>0.094925</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>303.0</td>
      <td>27.0</td>
      <td>0.081818</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>37.0</td>
      <td>4.0</td>
      <td>0.097561</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Kesimpulan**

Kesimpulan dari hipotesis ini adalah :
Nasabah yang tidak mempunyai anak memiliki probabilitas atau persentase dapat mengembalikan pinjaman tepat waktu dibandingkan dengan nasabah yang memiliki anak lebih dari satu.  

**Apakah terdapat korelasi antara status keluarga dengan membayar kembali tepat waktu?**


```python
# Periksa data status keluarga dan membayar kembali dengan tepat waktu
# Menghitung tarif otomatis berdasarkan status keluarga
df2 = df.pivot_table(index='family_status',columns='debt_meaning', values='debt', aggfunc='count') 
df2 = df2.reset_index()
df2['percentage'] = df2['Gagal Melunasi Pinjaman'] / (df2['Gagal Melunasi Pinjaman'] + df2['Berhasil Melunasi Pinjaman'])
df2
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
      <th>debt_meaning</th>
      <th>family_status</th>
      <th>Berhasil Melunasi Pinjaman</th>
      <th>Gagal Melunasi Pinjaman</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>civil partnership</td>
      <td>3763</td>
      <td>388</td>
      <td>0.093471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>divorced</td>
      <td>1110</td>
      <td>85</td>
      <td>0.071130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>married</td>
      <td>11408</td>
      <td>931</td>
      <td>0.075452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>unmarried</td>
      <td>2536</td>
      <td>274</td>
      <td>0.097509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>widow / widower</td>
      <td>896</td>
      <td>63</td>
      <td>0.065693</td>
    </tr>
  </tbody>
</table>
</div>



**Kesimpulan**


Kesimpulan dari hipotesis ini adalah :
Probabilitas dari persentase Nasabah dengan status keluarga widow/widower dapat mengembalikan pinjaman tepat waktu dibandingkan dengan status keluarga lainnya.


**Apakah terdapat korelasi antara tingkat pendapatan dengan membayar kembali tepat waktu?**




```python
# Periksa data tingkat pendapatan dan membayar kembali dengan tepat waktu
# Menghitung tarif otomatis berdasarkan tingkat pendapatan
df3 = df.pivot_table(index='level_income',columns='debt_meaning', values='debt', aggfunc='count') 
df3 = df3.reset_index()
df3['percentage'] = df3['Gagal Melunasi Pinjaman'] / (df3['Gagal Melunasi Pinjaman'] + df3['Berhasil Melunasi Pinjaman'])
df3
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
      <th>debt_meaning</th>
      <th>level_income</th>
      <th>Berhasil Melunasi Pinjaman</th>
      <th>Gagal Melunasi Pinjaman</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>High Income</td>
      <td>17758</td>
      <td>1587</td>
      <td>0.082037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lower Middle Income</td>
      <td>8</td>
      <td>1</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Upper Middle Income</td>
      <td>1947</td>
      <td>153</td>
      <td>0.072857</td>
    </tr>
  </tbody>
</table>
</div>



**Kesimpulan**

[Tulis kesimpulan Anda berdasarkan manipulasi dan pengamatan Anda.]

Dari Hasil hipotesis ini adalah :
Probabilitas dari persentasi nasabah dengan tingkat pendapatan menengah keatas dapat mengembalikan pinjaman tepat waktu dibandingkan dengan dengan tingkat pendapatan kelas atas. 

**Bagaimana tujuan kredit memengaruhi tarif otomatis?**


```python
# Periksa persentase tarif otomatis untuk setiap tujuan kredit dan lakukan penganalisisan
df4 = df.pivot_table(index='purpose',columns='debt_meaning', values='debt', aggfunc='count') 
df4 = df4.reset_index()
df4['percentage'] = df4['Gagal Melunasi Pinjaman'] / (df4['Gagal Melunasi Pinjaman'] + df4['Berhasil Melunasi Pinjaman'])
df4
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
      <th>debt_meaning</th>
      <th>purpose</th>
      <th>Berhasil Melunasi Pinjaman</th>
      <th>Gagal Melunasi Pinjaman</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Building Property Purpose</td>
      <td>2306</td>
      <td>179</td>
      <td>0.072032</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Buy Car Purpose</td>
      <td>3903</td>
      <td>403</td>
      <td>0.093590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Buy Property Purpose</td>
      <td>7723</td>
      <td>603</td>
      <td>0.072424</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Education Purpose</td>
      <td>3643</td>
      <td>370</td>
      <td>0.092200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wedding Purpose</td>
      <td>2138</td>
      <td>186</td>
      <td>0.080034</td>
    </tr>
  </tbody>
</table>
</div>



**Kesimpulan**

[Tulis kesimpulan Anda berdasarkan manipulasi dan pengamatan Anda.]

Dari Hasil hipotesis ini adalah : 
Probabilitas dari persentase nasabah dengan tujuan membangun property dan membeli property dapat mengembalikan pinjaman tepat waktu lebih rendah dibandingkan dengan nasabah yang memiliki tujuan pinjaman pernikahan, pendidikan dan membeli mobil. 

# Kesimpulan Umum 

[Tuliskan kesimpulan Anda di bagian akhir ini. Pastikan Anda memasukkan semua kesimpulan penting yang telah Anda buat berkaitan dengan cara Anda memproses dan menganalisis data. Mengatasi nilai yang hilang, duplikat, dan kemungkinan alasan serta solusi untuk data lama yang bermasalah yang harus Anda tangani.]

Kesimpulan Akhir :
1. Loading Data and Libraries
Langkah awal dalam memproses data yakni dengan mengimport library, memuat data lalu menampilkan informasi secara keseluruhan.

2. Explorasi Data / Data Quality Checking  
langkah kedua melakukan analisis data secara global dengan mengamati dengan detail value dari semua kolom serta tipe datanya. lalu memeriksa kualitas dari data tersebut yang meliputi. 
a. Memfilter value pada column yang terindikasi nilai yang hilang apakah terdapat nilai yang hilang secara acak atau terpola. 
b. Memeriksa duplikasi data pada datasets. 
c. Memeriksa nilai yang unique pada datasets apakah terdapat gaya penulisan yang tidak sesuai dengan formatnya.
d. Memeriksa Value apakah terdapat data yang tidak wajar sesuai dengan karakteristik pada value column tersebut.

3. Data Cleansing / Transformasi data
langkah ketiga melakukan pembersihan data atau memanipulasi data terhadap data yang terindikasi :
a. data yang terindikasi dengan gaya penulisan yang berbeda tetapi memiliki defenisi yang sama. dengan mereplace value yang lama dengan value yang baru.
b. data yang terindikasi tidak wajar terhadap realita pada value column tersebut contoh pada column 'children' memiliki value yang tidak wajar, 'education' memiliki gaya penulisan yang berbeda, 'days_employed' memiliki nilai yang negatif dan nilai yang tidak wajar, 'dob_years' memiliki nilai yang tidak wajar. 

Pada tahap ini kita mentransformasi nilai pada column 'children' dengan melakukan metode `.replace()` pada nilai yang terindikasi tidak wajar. lalu kita beralih pada column 'education' yang terindikasi gaya penulisan yang berbeda tetapi dengan defenisi yang sama, metode yang digunakan adalah melihat nilai yang `.unique()` lalu disamakan gaya penulisannya dengan metode `str.lower()`. beralih pada column 'days_employed' dimana terindikasi memiliki nilai negatif dengan merubah nilai tersebut menjadi positif dengan menggunakan metode `abs` serta menerapkan value yang nilainya terlalu tinggi dan merubahnya dengan nilai yang wajar berdasarkan nilai mean pada distribusi column tersebut. lalu beralih ke column 'dob_years'  merubah nilai yang tidak wajar dengan menerapkan nilai rata-rata pada 'dob_years'.

c. data yang memiliki nilai yang hilang pada column 'total_income' dan 'days_employed'.
    
 Pada tahap ini kita bekerja dengan nilai yang hilang pada column 'total_income' dengan membuat fungsi *def* berdasarkan kelompok umur dari lama nasabah bekerja lalu membuat column baru dengan nama 'age_group', yang nantinya value ini akan digunakan untuk menentukan nilai mean dan median guna mengisi nilai yang hilang. lalu membuat table / datasets baru yang tidak memiliki nilai yang hilang dengan metode mengeliminasi nilai yang hilang yakni *~.isna()* guna mencari nilai mean dan median untuk diterapkan pada datasets awal yang memiliki nilai yang hilang secara terpola. setelah mendapatkan nilainya disini saya menggunakan nilai median untuk mengisi nilai yang hilang ke column 'total_income' dengan metode *fillna*, dengan pertimbangan bahwa data memiliki nilai *OUTLIER* atau nilai yang terlalu tinggi dibandingkan denga rata-rata nilai yang lain. 
    Beralih kepada column 'days_employed' dimana terdapat nilai yang hilang, dengan metode mencari nilai rata-rata dan median berdasarkan column 'age_group'. setelah mendapatkan nilainya disini saya menggunakan nilai median untuk mengisi nilai yang hilang ke column 'total_income' dengan metode *fillna*, dengan pertimbangan bahwa data memiliki nilai *OUTLIER* atau nilai yang terlalu tinggi dibandingkan denga rata-rata nilai yang lain. 

4. Categorizing Data / Pengkategorian Data 
langkah keempat ini adalah mengkategorikan data sesuai dengan tipe data yang nantinya akan digunakan untuk menjawab hipotesis. 

a. Langkah pertama mengkategorikan data dengan tipe data kategoris, disini pengkategorian data menggunakan metode *pivot_table* dalam memilih tipe data kategoris untuk melihat distribusi data guna menjawab hipotesis. 

b. Langkah kedua mengkategorikan data dengan tipe data numerik, disini data menggunakan metode *pivot_table* dalam memilih tipe data numerik untuk melihat distribusi data guna menjawab hipotesis.

c. Langkah ketiga melihat distribusi nilai statistik atau rentang nilia  dari value pengkategorian tipe data kategoris dan numerik untuk mendapatkan insight atau kesimpulan untuk menjawab hipotesis.


[Tuliskan kesimpulan Anda mengenai pertanyaan yang ingin Anda ajukan di sini juga.]

5. Hipotesis / Conclusion 'Kesimpulan Akhir'

a. Apakah ada korelasi antara memiliki anak dengan ketepatan waktu dalam melunasi pinjaman?
    Dari hasil conclusion, didapatlah kesimpulan analisa dalam distribusi data sebagai berikut:
Kesimpulan dari hipotesis ini adalah : Nasabah yang tidak mempunyai anak memiliki probabilitas atau persentase dapat mengembalikan pinjaman tepat waktu dibandingkan dengan nasabah yang memiliki anak lebih dari satu.

b. Apakah ada korelasi antara status keluarga dengan ketepatan waktu dalam melunasi pinjaman?
    Dari hasil conclusion didapatlah kesimpulan analisa dalam distribusi data family status terhadap pembayaran hutang sebagai berikut:
Kesimpulan dari hipotesis ini adalah : Probabilitas dari persentase Nasabah dengan status keluarga widow/widower dapat mengembalikan pinjaman tepat waktu dibandingkan dengan status keluarga lainnya.

c. Apakah ada korelasi antara tingkat pendapatan dengan ketepatan waktu dalam melunasi pinjaman? 
    Dari hasil conclusion, didapatlah kesimpulan analisa dalam distribusi data level income terhadap pembayaran hutang sebagai berikut:
Dari Hasil hipotesis ini adalah : Probabilitas dari persentasi nasabah dengan tingkat pendapatan menengah keatas dapat mengembalikan pinjaman tepat waktu dibandingkan dengan dengan tingkat pendapatan kelas atas.

d. Bagaimana tujuan pinjaman mempengaruhi ketepatan waktu dalam melunasi pinjaman?
    Dari hasil conclusi, didapatlah kesimpulan analisa dalam distribusi data level income terhadap pembayaran hutang sebagai berikut:
Dari Hasil hipotesis ini adalah : Probabilitas dari persentase nasabah dengan tujuan membangun property dan membeli property dapat mengembalikan pinjaman tepat waktu lebih rendah dibandingkan dengan nasabah yang memiliki tujuan pinjaman pernikahan, pendidikan dan membeli mobil.




```python
# Mengekspor data akhir ke dalam pivot table
data_final = df
data_final = data_final.pivot_table(index=['days_employed','dob_years','children'], columns='debt', values='total_income', aggfunc='mean')
data_final
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
      <th>debt</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>days_employed</th>
      <th>dob_years</th>
      <th>children</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24.141633</th>
      <th>31</th>
      <th>1</th>
      <td>NaN</td>
      <td>26712.386</td>
    </tr>
    <tr>
      <th>24.240695</th>
      <th>32</th>
      <th>0</th>
      <td>19858.460000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30.195337</th>
      <th>47</th>
      <th>2</th>
      <td>37033.790000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33.520665</th>
      <th>43</th>
      <th>0</th>
      <td>NaN</td>
      <td>20568.944</td>
    </tr>
    <tr>
      <th>34.701045</th>
      <th>31</th>
      <th>1</th>
      <td>14489.279000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">14610.000000</th>
      <th rowspan="2" valign="top">72</th>
      <th>0</th>
      <td>18025.993240</td>
      <td>12668.922</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18942.954667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">73</th>
      <th>0</th>
      <td>16575.455000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18471.391000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <th>0</th>
      <td>11108.277750</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>16119 rows × 2 columns</p>
</div>


