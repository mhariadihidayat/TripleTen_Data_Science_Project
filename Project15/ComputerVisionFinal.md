# Deskripsi Proyek
Suatu waralaba supermarket bernama Good Seed ingin mengetahui apakah Data Science dapat membantu mereka mematuhi hukum dengan memastikan bahwa mereka tidak menjual produk yang memiliki batasan usia kepada pelanggan di bawah umur. Anda pun diminta untuk melaksanakan evaluasi. Oleh karena itu, saat Anda mulai bekerja, ingatlah hal-hal berikut ini:
Toko-toko dari waralaba ini dilengkapi dengan kamera di area kasir yang akan menampilkan sinyal ketika seseorang membeli produk dengan batasan usia
Metode visi komputer bisa digunakan untuk menentukan usia seseorang dari foto
Tugas Anda adalah membangun dan mengevaluasi sebuah model untuk memverifikasi usia seseorang
Untuk mulai mengerjakan tugas ini, Anda akan mendapatkan satu set foto orang dengan keterangan usianya.

## Inisialisasi 


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


print('Inisiasi Library Berhasil!!!')
```

    Inisiasi Library Berhasil!!!


## Memuat Data

*Dataset* yang Anda perlukan disimpan di folder `/datasets/faces/`. Pada folder tersebut, Anda bisa menemukan: - Folder `final_file` dengan 7,6 ribu foto 
- *File* `labels.csv` yang memuat label, dengan dua kolom: `file_name` dan `real_age` 

Mengingat jumlah *file* gambar cukup banyak, Anda disarankan untuk tidak membacanya sekaligus, karena hal ini hanya akan menghabiskan sumber daya komputasi. Kami sarankan Anda untuk membuat generator dengan ImageDataGenerator. Metode ini telah dijelaskan sebelumnya di Bab 3, Pelajaran ke-7. 

*File* label bisa dimuat sebagai *file* CSV biasa.


```python
# load dataset 
labels = pd.read_csv('/datasets/faces/labels.csv')
```


```python
labels.head()
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
      <th>file_name</th>
      <th>real_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000000.jpg</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000001.jpg</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000002.jpg</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000003.jpg</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000004.jpg</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
labels.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7591 entries, 0 to 7590
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   file_name  7591 non-null   object
     1   real_age   7591 non-null   int64 
    dtypes: int64(1), object(1)
    memory usage: 118.7+ KB



```python
labels.shape
```




    (7591, 2)




```python
labels.describe()
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
      <th>real_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7591.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>31.201159</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.145060</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>



Kesimpulan

Dari tampilan awal data, kita dapat melihat bahwa data tersebut memiliki 7591 baris dan 2 kolom. Dari gambaran data dapat diketahui bahwa usia rata-rata adalah 31 tahun dan usia maksimal adalah 100 tahun.

## EDA


```python
# data generator
train_datagen = ImageDataGenerator(rescale=1./255)

train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345) 
```

    Found 7591 validated image filenames.



```python
# gamabar dari perbedaan usia pelanggan 
features, target = next(train_gen_flow)
fig = plt.figure(figsize=(10,10))
for i in range(16):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(features[i])
    plt.title(target[i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
```


    
![png](output_13_0.png)
    



```python
# plot usia berdasarkan kelompok umur
bins = np.arange(labels['real_age'].min(), labels['real_age'].max()+6, 8)
bin_label = ['0 - 9', '9 - 17', '17 - 25', '25 - 33', '33 - 41', '41 - 49', '49 - 57',\
             '57 - 65', '65 - 73', '73 - 81', '81 - 89', '89 - 97', '97 - 100']
plot_bar = labels.groupby(pd.cut(labels['real_age'], bins=bins)).agg({'real_age': 'count'}).rename(columns={'real_age': 'Real age'})
plot_bar['Age groups'] = bin_label
fig, ax=plt.subplots(figsize=(12,6))
ax = sns.barplot(x='Age groups', y= 'Real age', data = plot_bar, edgecolor='black', hatch='/')
ax.set_title('Plot of real age by age group', fontdict={'size':12})
ax.set_xticklabels(bin_label, rotation=45);
for i, v in enumerate(plot_bar.iloc[:,0].values):
    ax.text(i + 0.25, v + 3, str(v), color='dodgerblue', fontweight='bold', fontdict={'horizontalalignment':'right', 'verticalalignment':'bottom', 'size':10})

```


    
![png](output_14_0.png)
    



```python
# distribusi data berdasarkan usia 
plt.figure(figsize=(12, 6))
sns.distplot(labels['real_age'], kde=True, bins=100, color='blue')
plt.title('Distribusi Usia')
plt.ylabel('Distribusi Data')
plt.xlabel('Rentang Usia')
plt.show()
```

    /opt/conda/lib/python3.9/site-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)



    
![png](output_15_1.png)
    


### Temuan

Dapat dilihat bahwa usia rata-rata pelanggan toko tersebut adalah sekitar 30 tahun. Kebanyakan orang berusia 17 - 41 tahun paling sering mengunjungi toko. Ada lebih sedikit orang tua yang mengunjungi toko. Tingginya jumlah anak usia 0 - 9 tahun yang mengunjungi toko tersebut kemungkinan karena beberapa orang tua mengunjungi toko tersebut bersama anaknya.

## Pemodelan 

Definisikan fungsi-fungsi yang diperlukan untuk melatih model Anda pada platform GPU dan buat satu skrip yang berisi semua fungsi tersebut beserta bagian inisialisasi.

Untuk mempermudah tugas ini, Anda dapat mendefinisikannya dalam *notebook* ini dan menjalankan kode siap pakai di bagian berikutnya untuk menyusun skrip secara otomatis.
Definisi di bawah ini juga akan diperiksa oleh *project reviewer* agar mereka dapat memahami cara Anda membangun model.


```python
def load_train(path):

    """
    Kode ini memuat bagian training set dari file path
    """

    # letakkan kode Anda di sini
    labels = pd.read_csv('/datasets/faces/labels.csv')
    # data generator
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=90)

    # ekstraksi data dari direktori
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345)

    return train_gen_flow
```


```python
def load_test(path):
    
    """
    Kode ini memuat bagian validation set/test set dari file path
    """
    
    # letakkan kode Anda di sini

    return test_gen_flow
```


```python
def load_test(path):

    """
    Kode ini memuat bagian validation set/test set dari file path
    """

    # letakkan kode Anda di sini
    labels = pd.read_csv('/datasets/faces/labels.csv')

    # validasi data generator
    test_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255)

    # ekstraksi data dari direktori
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)

    return test_gen_flow
```


```python
def create_model(input_shape):

    """
    Kode ini mendefinisikan model
    """

    # letakkan kode Anda di sini
    backbone = ResNet50(
        input_shape=input_shape, weights='imagenet', include_top=False
    )

    # model
    model = Sequential()

    # model layers 
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(350, activation='relu'))
    model.add(Dense(185, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # compiler
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['mae']
                 )


    return model
```


```python
def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=10,
    steps_per_epoch=None, validation_steps=None
):

    """
    Melatih model dengan parameter yang diberikan
    """

    # letakkan kode Anda di sini
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        verbose=2,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    return model
```

## Siapkan Skrip untuk menjalankan platform GPU

Setelah Anda mendefinisikan fungsi-fungsi yang diperlukan, Anda dapat membuat skrip untuk platform GPU, mengunduhnya melalui menu "File|Open...", dan mengunggahnya nanti untuk dijalankan pada platform GPU. "

Catatan: Skrip Anda juga harus menyertakan bagian inisialisasi. Contohnya ditunjukkan di bawah ini.


```python
# siapkan skrip untuk menjalankan platform GPU


init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:
    
    f.write(init_str)
    f.write('\n\n')
        
    for fn_name in [load_train, load_test, create_model, train_model]:
        
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')
```

### Output

Letakkan *output* dari platform GPU sebagai sel *Markdown* di sini.

`Found 5694 validated image filenames.`

`Found 1897 validated image filenames.`

`Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`

`<class 'tensorflow.python.keras.engine.sequential.Sequential'>
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
Train for 178 steps, validate for 60 steps
Epoch 1/10
2023-07-26 07:06:02.169623: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2023-07-26 07:06:02.977639: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
178/178 - 108s - loss: 240.5661 - mae: 11.6273 - val_loss: 305.8094 - val_mae: 13.0091
Epoch 2/10
178/178 - 93s - loss: 143.2351 - mae: 9.1596 - val_loss: 441.5143 - val_mae: 15.9408
Epoch 3/10
178/178 - 94s - loss: 123.4761 - mae: 8.5054 - val_loss: 280.8125 - val_mae: 12.4630
Epoch 4/10
178/178 - 93s - loss: 110.0389 - mae: 7.9904 - val_loss: 248.7191 - val_mae: 11.9805
Epoch 5/10
178/178 - 93s - loss: 101.8094 - mae: 7.7311 - val_loss: 127.7897 - val_mae: 8.4051
Epoch 6/10
178/178 - 93s - loss: 87.6314 - mae: 7.1397 - val_loss: 156.3676 - val_mae: 9.1996
Epoch 7/10
178/178 - 93s - loss: 82.6272 - mae: 6.9308 - val_loss: 121.9670 - val_mae: 8.2927
Epoch 8/10
178/178 - 94s - loss: 78.3582 - mae: 6.8130 - val_loss: 125.2969 - val_mae: 8.4183
Epoch 9/10
178/178 - 94s - loss: 69.1447 - mae: 6.4018 - val_loss: 92.4336 - val_mae: 7.1624
Epoch 10/10
178/178 - 93s - loss: 64.7149 - mae: 6.1700 - val_loss: 93.5149 - val_mae: 7.2643
WARNING:tensorflow:sample_weight modes were coerced from
   ...
    to  
  ['...']
60/60 - 9s - loss: 93.5149 - mae: 7.2643
Test MAE: 7.2643`

## Kesimpulan

Kami melakukan augmentasi pada set pelatihan (train_datagen) dengan menggunakan operasi berikut:

1. Refleksi vertikal dan horizontal
2. Rotasi hingga 90 derajat
3. Pergeseran vertikal dan horizontal gambar hingga 20% dari ukuran aslinya.

Kami membuat generator untuk set kereta dan pengujian, dan memuat data dari direktori. 
Kami membuat model menggunakan arsitektur ResNet50 dari TensorFlow. Kami melatih model dan menjalankan skrip pada platform GPU. Hasil keluaran dari platform GPU ditunjukkan di atas. Kami mengamati bahwa model tidak overfitting karena kerugian dan mean absolute error berkurang pada set pelatihan dan pengujian. MAE yang tercatat pada set tes adalah 7,5068.

Menggunakan kumpulan data dengan foto orang, kami dapat membangun dan melatih jaringan saraf convolutional pada platform GPU dengan skor MAE kurang dari 8,0. Metode computer vision yang dikembangkan ini dapat digunakan untuk menentukan umur seseorang dari sebuah foto. Model ini akan membantu jaringan supermarket Good Seed dengan membantu memverifikasi usia orang untuk mematuhi undang-undang alkohol dan tidak menjual alkohol kepada orang di bawah umur.

Tugas lain yang dapat diselesaikan dengan model tersebut termasuk deteksi penipuan kartu kredit. Rantai supermarket dapat menerapkan visi komputer dalam pengisian ulang otomatis dengan menangkap data gambar toko dan melakukan pemindaian inventaris lengkap dengan melacak item di rak dengan interval beberapa milidetik. Aplikasi lain akan berada di autonomous-checkout dimana sistem berbasis komputer dapat memahami interaksi pelanggan dan memantau pergerakan produk.

# Daftar Periksa

- [ ]  *Notebook* dibuka 
- [ ]  Tidak ada kesalahan dalam kode 
- [ ]  Sel dengan kode telah disusun berdasarkan urutan eksekusi 
- [ ]  Analisis data eksploratif telah dijalankan - [ ]  Hasil dari analisis data eksploratif ditampilkan pada *notebook* final - [ ]  Skor MAE model tidak lebih tinggi dari 8 
- [ ]  Kode pelatihan model telah disalin ke *notebook* final 
- [ ]  *Output* pelatihan model telah disalin di *notebook* final 
- [ ]  Temuan telah diberikan berdasarkan hasil pelatihan model 


```python

```
