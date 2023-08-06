# Deskripsi Proyek

Film Junky Union, sebuah komunitas baru bagi penggemar film klasik sedang mengembangkan sistem untuk memfilter dan mengategorikan ulasan film. Misi utamanya adalah melatih model agar bisa mendeteksi ulasan negatif secara otomatis. Anda akan menggunakan dataset ulasan film IMBD dengan pelabelan polaritas untuk membuat sebuah model yang bisa mengklasifikasikan ulasan positif dan negatif. Model ini setidaknya harus memiliki skor F1 sebesar 0,85.

# Instruksi Proyek
- Muat datanya.
- Lakukan pra-pemrosesan data apabila memang diperlukan.
- Lakukan EDA dan buat kesimpulan terkait ketidakseimbangan kelas.
- Lakukan pra-pemrosesan data untuk membuat model.
- Latih setidaknya tiga model untuk train dataset yang ada.
- Uji model untuk test dataset yang ada.
- Tulis beberapa ulasan Anda sendiri dan klasifikasikan dengan semua model.
- Periksa perbedaan antara hasil pengujian model dari dua poin di atas. Cobalah untuk menjelaskan hasilnya.
- Tampilkan hasil penemuan Anda.

## Inisialisasi


```python
import math
import re
import nltk
import spacy
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tqdm.auto import tqdm

# import stopword
! pip3 install wordcloud
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

# import train_test_split untuk membagi data
from sklearn.model_selection import train_test_split

# import library dari metrics
from sklearn import metrics
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, roc_curve)

import sys
import warnings 
warnings.filterwarnings("ignore")
```

    Collecting wordcloud
      Downloading wordcloud-1.9.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (460 kB)
    [K     |████████████████████████████████| 460 kB 14.9 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pillow in /opt/conda/lib/python3.9/site-packages (from wordcloud) (8.4.0)
    Requirement already satisfied: matplotlib in /opt/conda/lib/python3.9/site-packages (from wordcloud) (3.3.4)
    Requirement already satisfied: numpy>=1.6.1 in /opt/conda/lib/python3.9/site-packages (from wordcloud) (1.21.1)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.9/site-packages (from matplotlib->wordcloud) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /opt/conda/lib/python3.9/site-packages (from matplotlib->wordcloud) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.9/site-packages (from matplotlib->wordcloud) (1.4.2)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/lib/python3.9/site-packages (from matplotlib->wordcloud) (2.8.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.9/site-packages (from python-dateutil>=2.1->matplotlib->wordcloud) (1.16.0)
    Installing collected packages: wordcloud
    Successfully installed wordcloud-1.9.2



```python
# Time Progress 

!pip install ipython-autotime

%load_ext autotime
```

    Collecting ipython-autotime
      Downloading ipython_autotime-0.3.1-py2.py3-none-any.whl (6.8 kB)
    Requirement already satisfied: ipython in /opt/conda/lib/python3.9/site-packages (from ipython-autotime) (7.25.0)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (0.2.0)
    Requirement already satisfied: matplotlib-inline in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (0.1.2)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (2.9.0)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (4.8.0)
    Requirement already satisfied: traitlets>=4.2 in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (5.0.5)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (0.18.0)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (0.7.5)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (5.0.9)
    Requirement already satisfied: setuptools>=18.5 in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (49.6.0.post20210108)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.9/site-packages (from ipython->ipython-autotime) (3.0.19)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/conda/lib/python3.9/site-packages (from jedi>=0.16->ipython->ipython-autotime) (0.8.2)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.9/site-packages (from pexpect>4.3->ipython->ipython-autotime) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.9/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->ipython-autotime) (0.2.5)
    Requirement already satisfied: ipython-genutils in /opt/conda/lib/python3.9/site-packages (from traitlets>=4.2->ipython->ipython-autotime) (0.2.0)
    Installing collected packages: ipython-autotime
    Successfully installed ipython-autotime-0.3.1
    time: 373 µs (started: 2023-08-06 16:21:20 +00:00)



```python
%matplotlib inline
%config InlineBackend.figure_format = 'png'

# baris berikutnya menyediakan grafik dengan kualitas yang lebih baik di layar HiDPI 
%config InlineBackend.figure_format = 'retina'

plt.style.use('seaborn')
```

    time: 22.1 ms (started: 2023-08-06 16:21:20 +00:00)



```python
# ini untuk menggunakan progress_apply, baca lebih lanjut di https://pypi.org/project/tqdm/# pandas-integration
tqdm.pandas()
```

    time: 2.35 ms (started: 2023-08-06 16:21:20 +00:00)


## Memuat data


```python
df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})
```

    time: 1.09 s (started: 2023-08-06 16:21:20 +00:00)



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
    display(df.head(50))
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

    time: 3.51 ms (started: 2023-08-06 16:21:21 +00:00)



```python
get_info(df_reviews)
```

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
      <th>tconst</th>
      <th>title_type</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>end_year</th>
      <th>runtime_minutes</th>
      <th>is_adult</th>
      <th>genres</th>
      <th>average_rating</th>
      <th>votes</th>
      <th>review</th>
      <th>rating</th>
      <th>sp</th>
      <th>pos</th>
      <th>ds_part</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0068152</td>
      <td>movie</td>
      <td>$</td>
      <td>$</td>
      <td>1971</td>
      <td>\N</td>
      <td>121</td>
      <td>0</td>
      <td>Comedy,Crime,Drama</td>
      <td>6.3</td>
      <td>2218</td>
      <td>The pakage implies that Warren Beatty and Gold...</td>
      <td>1</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>8335</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0068152</td>
      <td>movie</td>
      <td>$</td>
      <td>$</td>
      <td>1971</td>
      <td>\N</td>
      <td>121</td>
      <td>0</td>
      <td>Comedy,Crime,Drama</td>
      <td>6.3</td>
      <td>2218</td>
      <td>How the hell did they get this made?! Presenti...</td>
      <td>1</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>8336</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0313150</td>
      <td>short</td>
      <td>'15'</td>
      <td>'15'</td>
      <td>2002</td>
      <td>\N</td>
      <td>25</td>
      <td>0</td>
      <td>Comedy,Drama,Short</td>
      <td>6.3</td>
      <td>184</td>
      <td>There is no real story the film seems more lik...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>2489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0313150</td>
      <td>short</td>
      <td>'15'</td>
      <td>'15'</td>
      <td>2002</td>
      <td>\N</td>
      <td>25</td>
      <td>0</td>
      <td>Comedy,Drama,Short</td>
      <td>6.3</td>
      <td>184</td>
      <td>Um .... a serious film about troubled teens in...</td>
      <td>7</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>9280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0313150</td>
      <td>short</td>
      <td>'15'</td>
      <td>'15'</td>
      <td>2002</td>
      <td>\N</td>
      <td>25</td>
      <td>0</td>
      <td>Comedy,Drama,Short</td>
      <td>6.3</td>
      <td>184</td>
      <td>I'm totally agree with GarryJohal from Singapo...</td>
      <td>9</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>9281</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt0313150</td>
      <td>short</td>
      <td>'15'</td>
      <td>'15'</td>
      <td>2002</td>
      <td>\N</td>
      <td>25</td>
      <td>0</td>
      <td>Comedy,Drama,Short</td>
      <td>6.3</td>
      <td>184</td>
      <td>This is the first movie I've seen from Singapo...</td>
      <td>9</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>9282</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tt0313150</td>
      <td>short</td>
      <td>'15'</td>
      <td>'15'</td>
      <td>2002</td>
      <td>\N</td>
      <td>25</td>
      <td>0</td>
      <td>Comedy,Drama,Short</td>
      <td>6.3</td>
      <td>184</td>
      <td>Yes non-Singaporean's can't see what's the big...</td>
      <td>9</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>9283</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tt0035958</td>
      <td>movie</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>1943</td>
      <td>\N</td>
      <td>88</td>
      <td>0</td>
      <td>Drama,History,War</td>
      <td>6.1</td>
      <td>1240</td>
      <td>This true story of Carlson's Raiders is more o...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>9903</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tt0035958</td>
      <td>movie</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>1943</td>
      <td>\N</td>
      <td>88</td>
      <td>0</td>
      <td>Drama,History,War</td>
      <td>6.1</td>
      <td>1240</td>
      <td>Should have been titled 'Balderdash!' Little i...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>9905</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tt0035958</td>
      <td>movie</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>1943</td>
      <td>\N</td>
      <td>88</td>
      <td>0</td>
      <td>Drama,History,War</td>
      <td>6.1</td>
      <td>1240</td>
      <td>The movie 'Gung Ho!': The Story of Carlson's M...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>9904</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tt0035958</td>
      <td>movie</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>'Gung Ho!': The Story of Carlson's Makin Islan...</td>
      <td>1943</td>
      <td>\N</td>
      <td>88</td>
      <td>0</td>
      <td>Drama,History,War</td>
      <td>6.1</td>
      <td>1240</td>
      <td>After reading the reviews, it became obvious t...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>9906</td>
    </tr>
    <tr>
      <th>11</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>The whole movie seemed to suffer from poor edi...</td>
      <td>1</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3579</td>
    </tr>
    <tr>
      <th>12</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>I don't know what has happened to director Abe...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3581</td>
    </tr>
    <tr>
      <th>13</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>"R Xmas" peers into the lives of a middle clas...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3577</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>Good actors, good director, well acted, well d...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3578</td>
    </tr>
    <tr>
      <th>15</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>I caught the North American premiere of this a...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3582</td>
    </tr>
    <tr>
      <th>16</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>The whole movie was done half-assed. It could ...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3583</td>
    </tr>
    <tr>
      <th>17</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>In New York, in a morning close to Christmas, ...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3576</td>
    </tr>
    <tr>
      <th>18</th>
      <td>tt0217978</td>
      <td>movie</td>
      <td>'R Xmas</td>
      <td>'R Xmas</td>
      <td>2001</td>
      <td>\N</td>
      <td>85</td>
      <td>0</td>
      <td>Crime,Drama,Thriller</td>
      <td>5.8</td>
      <td>1275</td>
      <td>'R Xmas is one of the only films I've seen whe...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3580</td>
    </tr>
    <tr>
      <th>19</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>I've seen Foxy Brown, Coffy, Friday Foster, Bu...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3354</td>
    </tr>
    <tr>
      <th>20</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>I really tried, but this movie just didn't wor...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3355</td>
    </tr>
    <tr>
      <th>21</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>When Pam Grier made COFFY in 1973, it was an e...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3356</td>
    </tr>
    <tr>
      <th>22</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>Sheba Shayne (Pam Grier) receives a telegram i...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3357</td>
    </tr>
    <tr>
      <th>23</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>Below average blaxpoitation action / melodrama...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3358</td>
    </tr>
    <tr>
      <th>24</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>If anything, William Girdler was an opportunis...</td>
      <td>7</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>12353</td>
    </tr>
    <tr>
      <th>25</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>Pistol-packing Pam Grier takes names and kicks...</td>
      <td>8</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>12352</td>
    </tr>
    <tr>
      <th>26</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>Sheba Baby is always underrated most likely be...</td>
      <td>8</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>12354</td>
    </tr>
    <tr>
      <th>27</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>Sheba Baby, is another Pam Grier Blaxploitatio...</td>
      <td>8</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>12355</td>
    </tr>
    <tr>
      <th>28</th>
      <td>tt0073697</td>
      <td>movie</td>
      <td>'Sheba, Baby'</td>
      <td>'Sheba, Baby'</td>
      <td>1975</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Action,Crime,Drama</td>
      <td>5.8</td>
      <td>1204</td>
      <td>Yeah, I guess this movie is kinda dull compare...</td>
      <td>10</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>12356</td>
    </tr>
    <tr>
      <th>29</th>
      <td>tt0118523</td>
      <td>movie</td>
      <td>'Til There Was You</td>
      <td>'Til There Was You</td>
      <td>1997</td>
      <td>\N</td>
      <td>113</td>
      <td>0</td>
      <td>Comedy,Romance</td>
      <td>4.8</td>
      <td>2625</td>
      <td>'Til There was You is one of the worst films w...</td>
      <td>1</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3715</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0118523</td>
      <td>movie</td>
      <td>'Til There Was You</td>
      <td>'Til There Was You</td>
      <td>1997</td>
      <td>\N</td>
      <td>113</td>
      <td>0</td>
      <td>Comedy,Romance</td>
      <td>4.8</td>
      <td>2625</td>
      <td>From the beginning, 'Til There Was You was on ...</td>
      <td>1</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3716</td>
    </tr>
    <tr>
      <th>31</th>
      <td>tt0118523</td>
      <td>movie</td>
      <td>'Til There Was You</td>
      <td>'Til There Was You</td>
      <td>1997</td>
      <td>\N</td>
      <td>113</td>
      <td>0</td>
      <td>Comedy,Romance</td>
      <td>4.8</td>
      <td>2625</td>
      <td>The film has weird annoying characters, strang...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3717</td>
    </tr>
    <tr>
      <th>32</th>
      <td>tt0090556</td>
      <td>movie</td>
      <td>'night, Mother</td>
      <td>'night, Mother</td>
      <td>1986</td>
      <td>\N</td>
      <td>96</td>
      <td>0</td>
      <td>Drama</td>
      <td>7.6</td>
      <td>2070</td>
      <td>************* SPOILERS BELOW ************* "'N...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>6139</td>
    </tr>
    <tr>
      <th>33</th>
      <td>tt0090556</td>
      <td>movie</td>
      <td>'night, Mother</td>
      <td>'night, Mother</td>
      <td>1986</td>
      <td>\N</td>
      <td>96</td>
      <td>0</td>
      <td>Drama</td>
      <td>7.6</td>
      <td>2070</td>
      <td>I had the privilege of seeing this powerful pl...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>6140</td>
    </tr>
    <tr>
      <th>34</th>
      <td>tt0108999</td>
      <td>movie</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>1994</td>
      <td>\N</td>
      <td>99</td>
      <td>0</td>
      <td>Drama</td>
      <td>6.4</td>
      <td>126</td>
      <td>I rented this movie from a local library witho...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>3142</td>
    </tr>
    <tr>
      <th>35</th>
      <td>tt0108999</td>
      <td>movie</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>1994</td>
      <td>\N</td>
      <td>99</td>
      <td>0</td>
      <td>Drama</td>
      <td>6.4</td>
      <td>126</td>
      <td>The movie ". . . And The Earth Did not Swallow...</td>
      <td>8</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>11621</td>
    </tr>
    <tr>
      <th>36</th>
      <td>tt0108999</td>
      <td>movie</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>1994</td>
      <td>\N</td>
      <td>99</td>
      <td>0</td>
      <td>Drama</td>
      <td>6.4</td>
      <td>126</td>
      <td>I was very moved by the young life experiences...</td>
      <td>10</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>11622</td>
    </tr>
    <tr>
      <th>37</th>
      <td>tt0108999</td>
      <td>movie</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>...And the Earth Did Not Swallow Him</td>
      <td>1994</td>
      <td>\N</td>
      <td>99</td>
      <td>0</td>
      <td>Drama</td>
      <td>6.4</td>
      <td>126</td>
      <td>Recently finally available in DVD (11/11/08), ...</td>
      <td>10</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>11623</td>
    </tr>
    <tr>
      <th>38</th>
      <td>tt0636268</td>
      <td>tvEpisode</td>
      <td>...In Translation</td>
      <td>...In Translation</td>
      <td>2005</td>
      <td>\N</td>
      <td>43</td>
      <td>0</td>
      <td>Adventure,Drama,Fantasy</td>
      <td>8.5</td>
      <td>4247</td>
      <td>I know that in this episode there's other stuf...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>8630</td>
    </tr>
    <tr>
      <th>39</th>
      <td>tt0109000</td>
      <td>movie</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>1994</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Comedy</td>
      <td>7.1</td>
      <td>2431</td>
      <td>I'm sorry but I cannot even partly agree with ...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>test</td>
      <td>1357</td>
    </tr>
    <tr>
      <th>40</th>
      <td>tt0109000</td>
      <td>movie</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>1994</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Comedy</td>
      <td>7.1</td>
      <td>2431</td>
      <td>i consider this movie as one of the most inter...</td>
      <td>10</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>5270</td>
    </tr>
    <tr>
      <th>41</th>
      <td>tt0109000</td>
      <td>movie</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>1994</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Comedy</td>
      <td>7.1</td>
      <td>2431</td>
      <td>Helges best movie by far. Very funny, very sur...</td>
      <td>10</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>5271</td>
    </tr>
    <tr>
      <th>42</th>
      <td>tt0109000</td>
      <td>movie</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>00 Schneider - Jagd auf Nihil Baxter</td>
      <td>1994</td>
      <td>\N</td>
      <td>90</td>
      <td>0</td>
      <td>Comedy</td>
      <td>7.1</td>
      <td>2431</td>
      <td>The phenomenon Helge Schneider defies easy des...</td>
      <td>10</td>
      <td>pos</td>
      <td>1</td>
      <td>test</td>
      <td>5272</td>
    </tr>
    <tr>
      <th>43</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>While watching this movie, I came up with a sc...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1412</td>
    </tr>
    <tr>
      <th>44</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>I love Morgan Freeman. Paz Vega is an attracti...</td>
      <td>2</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1413</td>
    </tr>
    <tr>
      <th>45</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>After I watched this movie, I came to IMDb and...</td>
      <td>3</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1414</td>
    </tr>
    <tr>
      <th>46</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>Not really spoilers in my opinion, but I wante...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1409</td>
    </tr>
    <tr>
      <th>47</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>A still famous but decadent actor (Morgan Free...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1410</td>
    </tr>
    <tr>
      <th>48</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>This movie was great the first time I saw it, ...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1411</td>
    </tr>
    <tr>
      <th>49</th>
      <td>tt0499603</td>
      <td>movie</td>
      <td>10 Items or Less</td>
      <td>10 Items or Less</td>
      <td>2006</td>
      <td>\N</td>
      <td>82</td>
      <td>0</td>
      <td>Comedy,Drama</td>
      <td>6.6</td>
      <td>14062</td>
      <td>I had high expectations for this indie having ...</td>
      <td>4</td>
      <td>neg</td>
      <td>0</td>
      <td>train</td>
      <td>1415</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    Info:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 47331 entries, 0 to 47330
    Data columns (total 17 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   tconst           47331 non-null  object 
     1   title_type       47331 non-null  object 
     2   primary_title    47331 non-null  object 
     3   original_title   47331 non-null  object 
     4   start_year       47331 non-null  int64  
     5   end_year         47331 non-null  object 
     6   runtime_minutes  47331 non-null  object 
     7   is_adult         47331 non-null  int64  
     8   genres           47331 non-null  object 
     9   average_rating   47329 non-null  float64
     10  votes            47329 non-null  Int64  
     11  review           47331 non-null  object 
     12  rating           47331 non-null  int64  
     13  sp               47331 non-null  object 
     14  pos              47331 non-null  int64  
     15  ds_part          47331 non-null  object 
     16  idx              47331 non-null  int64  
    dtypes: Int64(1), float64(1), int64(5), object(10)
    memory usage: 6.2+ MB



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
      <th>start_year</th>
      <th>is_adult</th>
      <th>average_rating</th>
      <th>votes</th>
      <th>rating</th>
      <th>pos</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>47331.000000</td>
      <td>47331.000000</td>
      <td>47329.000000</td>
      <td>4.732900e+04</td>
      <td>47331.000000</td>
      <td>47331.000000</td>
      <td>47331.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1989.631235</td>
      <td>0.001732</td>
      <td>5.998278</td>
      <td>2.556292e+04</td>
      <td>5.484608</td>
      <td>0.498954</td>
      <td>6279.697999</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.600364</td>
      <td>0.041587</td>
      <td>1.494289</td>
      <td>8.367004e+04</td>
      <td>3.473109</td>
      <td>0.500004</td>
      <td>3605.702545</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1894.000000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>9.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1982.000000</td>
      <td>0.000000</td>
      <td>5.100000</td>
      <td>8.270000e+02</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3162.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1998.000000</td>
      <td>0.000000</td>
      <td>6.300000</td>
      <td>3.197000e+03</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>6299.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2004.000000</td>
      <td>0.000000</td>
      <td>7.100000</td>
      <td>1.397400e+04</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>9412.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2010.000000</td>
      <td>1.000000</td>
      <td>9.700000</td>
      <td>1.739448e+06</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>12499.000000</td>
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
      <th>tconst</th>
      <th>title_type</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>end_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>review</th>
      <th>sp</th>
      <th>ds_part</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
      <td>47331</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>6648</td>
      <td>10</td>
      <td>6555</td>
      <td>6562</td>
      <td>60</td>
      <td>249</td>
      <td>585</td>
      <td>47240</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>tt0116130</td>
      <td>movie</td>
      <td>The Sentinel</td>
      <td>The Sentinel</td>
      <td>\N</td>
      <td>90</td>
      <td>Drama</td>
      <td>Loved today's show!!! It was a variety and not...</td>
      <td>neg</td>
      <td>train</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>30</td>
      <td>36861</td>
      <td>60</td>
      <td>60</td>
      <td>45052</td>
      <td>2442</td>
      <td>3392</td>
      <td>5</td>
      <td>23715</td>
      <td>23796</td>
    </tr>
  </tbody>
</table>
</div>


    
    Columns dengan nilai yang hilang:
    Column average_rating dengan 0.0042% persentasi nilai yang hilang , dan 2 nilai yang hilang
    Column votes dengan 0.0042% persentasi nilai yang hilang , dan 2 nilai yang hilang
    [1mTerdapat 2 columns dengan nilai NA.[0m



    None


    ----------------------------------------------------------------------------------------------------
    Shape:
    (47331, 17)
    ----------------------------------------------------------------------------------------------------
    Duplicated:
    [1mKita mempunyai 0 baris yang terduplikasi.
    [0m
    
    time: 732 ms (started: 2023-08-06 16:21:21 +00:00)


## EDA

Memeriksa jumlah film dan ulasan selama beberapa tahun.


```python
fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Jumlah Film Selama Beberapa Tahun')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#ulasan  (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='ulasan per film (rata-rata selama 5 tahun)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Jumlah Ulasan Selama Beberapa Tahun') 

fig.tight_layout()
```


    
![png](output_13_0.png)
    


    time: 5.6 s (started: 2023-08-06 16:21:22 +00:00)


Memeriksa distribusi jumlah ulasan per film dengan penghitungan yang tepat dan KDE (hanya untuk mengetahui perbedaannya dari penghitungan yang tepat)


```python
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Plot batang #Ulasan Per Film') 

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('Plot KDE #Ulasan Per Film') 

fig.tight_layout()
```


    
![png](output_15_0.png)
    


    time: 1.18 s (started: 2023-08-06 16:21:27 +00:00)



```python
df_reviews['pos'].value_counts()
```




    0    23715
    1    23616
    Name: pos, dtype: int64



    time: 4.86 ms (started: 2023-08-06 16:21:29 +00:00)



```python
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('Train set: distribusi peringkat')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('Test set: distribusi peringkat')

fig.tight_layout()
```


    
![png](output_17_0.png)
    


    time: 629 ms (started: 2023-08-06 16:21:29 +00:00)


Plot Distribusi ulasan negatif dan positif selama bertahun-tahun untuk dua bagian *dataset*


```python
fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Train set: jumlah ulasan dari polaritas yang berbeda per tahun')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('Train set: distribusi dari polaritas yang berbeda per film')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Test set: jumlah ulasan dari polaritas yang berbeda per tahun')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('Test set: distribusi dari polaritas yang berbeda per film')

fig.tight_layout()
```


    
![png](output_19_0.png)
    


    time: 6.57 s (started: 2023-08-06 16:21:29 +00:00)


## Prosedur Evaluasi

Menyusun evaluasi yang dapat digunakan untuk semua model dalam tugas ini secara rutin


```python
import sklearn.metrics as metrics

def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Scor F1 
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # menetapkan persilangan untuk beberapa ambang batas
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Skor F1') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # menetapkan persilangan untuk beberapa ambang batas
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'Kurva ROC')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # menetapkan persilangan untuk beberapa ambang batas
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return
```

    time: 3.94 ms (started: 2023-08-06 16:21:36 +00:00)


## Normalisasi

Kita akan membuat fungsi untuk menormalisasi data dari text yang berfifat tidak standar seperti angka, tanda baca, dll.


```python
# fungsi untuk clean the data
def clean_data(data):
    ''' 
    fungsi ini digunakan untuk membersihkan
    data karakter non standar
    '''
    clean_data = re.sub(r"[^a-zA-Z']", " ", data)
    clean_data = " ".join(clean_data.split())
    return clean_data.lower()
```

    time: 1.23 ms (started: 2023-08-06 16:21:36 +00:00)



```python
df_reviews['review_norm'] = df_reviews['review'].apply(clean_data)
df_reviews['review_norm']
```




    0        the pakage implies that warren beatty and gold...
    1        how the hell did they get this made presenting...
    2        there is no real story the film seems more lik...
    3        um a serious film about troubled teens in sing...
    4        i'm totally agree with garryjohal from singapo...
                                   ...                        
    47326    this is another of my favorite columbos it spo...
    47327    talk about being boring i got this expecting a...
    47328    i never thought i'd say this about a biopic bu...
    47329    spirit and chaos is an artistic biopic of miya...
    47330    i'll make this brief this was a joy to watch i...
    Name: review_norm, Length: 47331, dtype: object



    time: 4.76 s (started: 2023-08-06 16:21:36 +00:00)


## Pemisahan Train / Test

Memisahkan seluruh *dataset* menjadi *train/test*. dimana kolom parameternya adalah 'ds_part'.


```python
df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

print(df_reviews_train.shape)
print(df_reviews_test.shape)
```

    (23796, 18)
    (23535, 18)
    time: 56.5 ms (started: 2023-08-06 16:21:41 +00:00)



```python
y_train = df_reviews_train['pos']
y_test = df_reviews_test['pos']

X_train = df_reviews_train['review_norm']
X_test = df_reviews_test['review_norm']

```

    time: 2.48 ms (started: 2023-08-06 16:21:41 +00:00)


## Bekerja dengan Model

### Model 1 - Menggunakan NLTK & TF-IDF 

Pada tahap ini kita akan mendefenisikan Term Frequency — Inverse Document Frequency untuk di mengkonversi data teks menjadi vector, tetapi sebelumnya kita akan melakukan proses lemmatization pada feature dimana proses ini membantu mempermudah penanganan teks karena variasi bentuk kata yang tersedia menjadi lebih sedikit.


```python
import nltk

from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
```

    time: 607 µs (started: 2023-08-06 16:21:41 +00:00)



```python
# fungsi membuat token dan lemmatiz text 
def lemmatize_nltk(text):
    lemmatizer  = WordNetLemmatizer() # membuat lemmatiz pada data object
    tokens = word_tokenize(text.lower()) # memisahkan text kedalam tokens
    lemmas = [lemmatizer.lemmatize(token) for token in tokens] 
    lemmatized = " ".join(lemmas) 
    return lemmatized
```

    time: 782 µs (started: 2023-08-06 16:21:41 +00:00)



```python
# text pre-processing dengan NLTK
stop_words = set(nltk_stopwords.words('english'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words) 
```

    time: 34.2 ms (started: 2023-08-06 16:21:41 +00:00)


####  Menjalankan fungsi token dan lemmatiz  pda train & test sampel


```python
# menjalankan fungsi token dan lemmatiz  pda train & test sampel
df_reviews_train['review_nltk'] = X_train.apply(lemmatize_nltk)
df_reviews_test['review_nltk'] = X_test.apply(lemmatize_nltk)

# menjalankan TF-IDF pada train & test sampel
nltk_X_train = count_tf_idf.fit_transform(df_reviews_train['review_nltk'])
nltk_X_test = count_tf_idf.transform(df_reviews_test['review_nltk'])
```

    time: 2min 8s (started: 2023-08-06 16:21:41 +00:00)


#### Data Frame hasil Model dengan TF IDF & NLTK


```python
# Data Frame hasil Model dengan TF IDF & NLTK 
log_metrics_tf_idf = {"Models": ["Logistic Regression Classifier", "Gaussian Naive Bayes", "LightGBM Classifier" ,"Stochastic Gradient Classifier" ], "Accuracy_score": [0.0]*4, "F1 Score": [0.0]*4}
m_idx_tf_idf = {"Logistic Regression Classifier":0, "Gaussian Naive Bayes":1, "LightGBM Classifier":2, "Stochastic Gradient Classifier" :3}
```

    time: 727 µs (started: 2023-08-06 16:23:50 +00:00)


#### Logistic Regression Classifier


```python
from sklearn.linear_model import LogisticRegression
```

    time: 1.01 ms (started: 2023-08-06 16:23:50 +00:00)



```python
%%time

lr = LogisticRegression(random_state=12345, solver='liblinear')
lr.fit(nltk_X_train, y_train)
lr_tf_idf_pred = lr.predict(nltk_X_test)
```

    CPU times: user 770 ms, sys: 1.41 s, total: 2.18 s
    Wall time: 2.18 s
    time: 2.19 s (started: 2023-08-06 16:23:50 +00:00)



```python
acc_lr = accuracy_score(y_test, lr_tf_idf_pred)
f1_sc_lr  = f1_score(y_test, lr_tf_idf_pred)

log_metrics_tf_idf["Accuracy_score"][m_idx_tf_idf["Logistic Regression Classifier"]] = acc_lr
log_metrics_tf_idf["F1 Score"][m_idx_tf_idf["Logistic Regression Classifier"]] = f1_sc_lr
```

    time: 14.9 ms (started: 2023-08-06 16:23:52 +00:00)



```python
evaluate_model(lr, nltk_X_train, y_train, nltk_X_test, y_test)
```

              train  test
    Accuracy   0.94  0.88
    F1         0.94  0.88
    APS        0.98  0.95
    ROC AUC    0.98  0.95



    
![png](output_45_1.png)
    


    time: 1.28 s (started: 2023-08-06 16:23:52 +00:00)


#### Gusion naïve Bayes Classifier 


```python
from sklearn.naive_bayes import MultinomialNB
```

    time: 3.88 ms (started: 2023-08-06 16:23:53 +00:00)



```python
%%time
gnb = MultinomialNB()
gnb.fit(nltk_X_train, y_train)
gnb_tf_idf_pred = gnb.predict(nltk_X_test)

```

    CPU times: user 44.3 ms, sys: 111 µs, total: 44.4 ms
    Wall time: 43.3 ms
    time: 44.8 ms (started: 2023-08-06 16:23:53 +00:00)



```python
acc_gnb = accuracy_score(y_test, gnb_tf_idf_pred)
f1_sc_gnb  = f1_score(y_test, gnb_tf_idf_pred)

log_metrics_tf_idf["Accuracy_score"][m_idx_tf_idf["Gaussian Naive Bayes"]] = acc_gnb
log_metrics_tf_idf["F1 Score"][m_idx_tf_idf["Gaussian Naive Bayes"]] = f1_sc_gnb
```

    time: 33 ms (started: 2023-08-06 16:23:53 +00:00)



```python
evaluate_model(gnb, nltk_X_train, y_train, nltk_X_test, y_test)
```

              train  test
    Accuracy   0.91  0.83
    F1         0.91  0.82
    APS        0.97  0.91
    ROC AUC    0.97  0.91



    
![png](output_50_1.png)
    


    time: 1.4 s (started: 2023-08-06 16:23:53 +00:00)


#### LightGBM Classifier


```python
from lightgbm import LGBMClassifier
```

    time: 25.9 ms (started: 2023-08-06 16:23:55 +00:00)



```python
%%time 

lgb = LGBMClassifier(random_state=12345)
lgb.fit(nltk_X_train, y_train)
lgb_tf_idf_pred = lgb.predict(nltk_X_test)
```

    CPU times: user 1min 5s, sys: 377 ms, total: 1min 5s
    Wall time: 1min 6s
    time: 1min 6s (started: 2023-08-06 16:23:55 +00:00)



```python
acc_lgb = accuracy_score(y_test, lgb_tf_idf_pred)
f1_sc_lgb  = f1_score(y_test, lgb_tf_idf_pred)

log_metrics_tf_idf["Accuracy_score"][m_idx_tf_idf["LightGBM Classifier"]] = acc_lgb
log_metrics_tf_idf["F1 Score"][m_idx_tf_idf["LightGBM Classifier"]] = f1_sc_lgb
```

    time: 12.5 ms (started: 2023-08-06 16:25:01 +00:00)



```python
evaluate_model(lgb, nltk_X_train, y_train, nltk_X_test, y_test)
```

              train  test
    Accuracy   0.92  0.86
    F1         0.92  0.86
    APS        0.97  0.93
    ROC AUC    0.97  0.94



    
![png](output_55_1.png)
    


    time: 2.51 s (started: 2023-08-06 16:25:01 +00:00)


#### Stochastic Gradient Classifier


```python
from sklearn.linear_model import SGDClassifier 
```

    time: 432 µs (started: 2023-08-06 16:25:03 +00:00)



```python
%%time

sgd = SGDClassifier(random_state=12345, loss='log')
sgd.fit(nltk_X_train, y_train)
sgd_tf_idf_pred = sgd.predict(nltk_X_test)
```

    CPU times: user 203 ms, sys: 128 ms, total: 331 ms
    Wall time: 334 ms
    time: 354 ms (started: 2023-08-06 16:25:03 +00:00)



```python
acc_sgd = accuracy_score(y_test, sgd_tf_idf_pred)
f1_sc_sgd  = f1_score(y_test, sgd_tf_idf_pred)

log_metrics_tf_idf["Accuracy_score"][m_idx_tf_idf["Stochastic Gradient Classifier"]] = acc_sgd
log_metrics_tf_idf["F1 Score"][m_idx_tf_idf["Stochastic Gradient Classifier"]] = f1_sc_sgd
```

    time: 12.9 ms (started: 2023-08-06 16:25:04 +00:00)



```python
evaluate_model(sgd, nltk_X_train, y_train, nltk_X_test, y_test)
```

              train  test
    Accuracy   0.92  0.88
    F1         0.92  0.88
    APS        0.97  0.95
    ROC AUC    0.97  0.95



    
![png](output_60_1.png)
    


    time: 1.34 s (started: 2023-08-06 16:25:04 +00:00)


### Model 2 - Menggunakan spaCy & TF-IDF 

Pada tahap ini kita akan menggunakan NLP Natural Languange Preprocessing dengan Proses ini membantu mempermudah penanganan teks karena variasi bentuk kata yang tersedia menjadi lebih sedikit.


```python
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
```


```python
# text pre-processing dengan spaCy
def lemmatize_spacy(text):
    doc = nlp(text.lower())
    lemmas = []
    #tokens = [token.lemma_ for token in doc if not token.is_stop]
    lemmas = [token.lemma_ for token in doc]
    lemmatize_spacy = " ".join(lemmas)
    return lemmatize_spacy
```

#### Menjalankan fungsi token dan lemmatiz pada train & test sampel spaCy


```python
# menjalankan fungsi token dan lemmatiz  pada train & test sampel spaCy
df_reviews_train['review_spacy'] = X_train.apply(lemmatize_spacy)
df_reviews_test['review_spacy'] = X_test.apply(lemmatize_spacy)

# menjalankan TF-IDF pada train & test sampel
spacy_X_train = count_tf_idf.fit_transform(df_reviews_train['review_spacy'])
spacy_X_test = count_tf_idf.transform(df_reviews_test['review_spacy'])
```

#### Data Frame hasil Model dengan spaCy & TF-IDF


```python
# Data Frame hasil Model dengan spaCy & TF-IDF 
log_metrics_spacy = {"Models": ["Logistic Regression Classifier", "Gaussian Naive Bayes", "LightGBM Classifier" ,"Stochastic Gradient Classifier" ], "Accuracy_score": [0.0]*4, "F1 Score": [0.0]*4}
m_idx_spacy = {"Logistic Regression Classifier":0, "Gaussian Naive Bayes":1, "LightGBM Classifier":2, "Stochastic Gradient Classifier" :3}
```

#### Logistic Regression 


```python
%%time

lr_spacy = LogisticRegression(random_state=12345, solver='liblinear')
lr_spacy.fit(spacy_X_train, y_train)
lr_spacy_pred = lr_spacy.predict(spacy_X_test)
```


```python
acc_lr_spacy = accuracy_score(y_test, lr_spacy_pred )
f1_sc_lr_spacy  = f1_score(y_test, lr_spacy_pred )

log_metrics_spacy["Accuracy_score"][m_idx_spacy["Logistic Regression Classifier"]] = acc_lr_spacy
log_metrics_spacy["F1 Score"][m_idx_spacy["Logistic Regression Classifier"]] = f1_sc_lr_spacy
```


```python
evaluate_model(lr_spacy, spacy_X_train, y_train, spacy_X_test, y_test)
```

#### Gusion naïve Bayes Classifier


```python
%%time
gnb_spacy = MultinomialNB()
gnb_spacy.fit(spacy_X_train, y_train)
gnb_spacy_pred = gnb_spacy.predict(spacy_X_test)

```


```python
acc_gnb_spacy = accuracy_score(y_test, gnb_spacy_pred)
f1_sc_gnb_spacy = f1_score(y_test, gnb_spacy_pred)

log_metrics_spacy["Accuracy_score"][m_idx_spacy["Gaussian Naive Bayes"]] = acc_gnb_spacy
log_metrics_spacy["F1 Score"][m_idx_spacy["Gaussian Naive Bayes"]] = f1_sc_gnb_spacy
```


```python
evaluate_model(gnb_spacy, spacy_X_train, y_train, spacy_X_test, y_test)
```

#### LightGBM Classifier


```python
%%time 

lgb_spacy = LGBMClassifier(random_state=12345)
lgb_spacy.fit(spacy_X_train, y_train)
lgb_spacy_pred = lgb_spacy.predict(spacy_X_test)
```


```python
acc_lgb_spacy = accuracy_score(y_test, lgb_spacy_pred)
f1_sc_lgb_spacy = f1_score(y_test, lgb_spacy_pred)

log_metrics_spacy["Accuracy_score"][m_idx_spacy["LightGBM Classifier"]] = acc_lgb_spacy
log_metrics_spacy["F1 Score"][m_idx_spacy["LightGBM Classifier"]] = f1_sc_lgb_spacy
```


```python
evaluate_model(lgb_spacy, spacy_X_train, y_train, spacy_X_test, y_test)
```

#### Stochastic Gradient Classifier


```python
%%time

sgd_spacy = SGDClassifier(random_state=12345, loss='log')
sgd_spacy.fit(spacy_X_train, y_train)
sgd_spacy_pred = sgd_spacy.predict(spacy_X_test)
```


```python
acc_sgd_spacy = accuracy_score(y_test, sgd_spacy_pred)
f1_sc_sgd_spacy  = f1_score(y_test, sgd_spacy_pred)

log_metrics_spacy["Accuracy_score"][m_idx_spacy["Stochastic Gradient Classifier"]] = acc_sgd_spacy
log_metrics_spacy["F1 Score"][m_idx_spacy["Stochastic Gradient Classifier"]] = f1_sc_sgd_spacy
```


```python
evaluate_model(sgd_spacy, spacy_X_train, y_train, spacy_X_test, y_test)
```

###  Model 3 - Menggunakan BERT

Pada tahap ini kita akan mengubah teks ke vektor menggunakan BERT (Bidirectional Encoder Representations from Transformers) adalah jaringan neural (neural network) yang dibuat untuk merepresentasikan bahasa yakni model Deep Learning untuk NLP yang didasarkan pada Transformer di mana setiap elemen output terhubung ke setiap elemen input, dan bobot elemen dihitung secara dinamis berdasarkan hubungan antar elemen.


```python
import torch
import transformers
```


```python
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')
```


```python
# maximum sample size
max_sample_size = 500
```


```python
# Fungsi untuk menjalankan bert
def BERT_text_to_embeddings(texts, max_length=512, batch_size=25, force_device=None, disable_progress_bar=False):
    
    ids_list = []
    attention_mask_list = []
    max_sample_size = 500
    
    # teks ke id token yang sudah di-padded bersamaan dengan attention mask
    for input_text in texts.iloc[:max_sample_size]['review']:
        ids = tokenizer.encode(input_text.lower(), add_special_tokens=True, truncation=True, max_length=max_length)
        padded = np.array(ids + [0]*(max_length - len(ids)))
        attention_mask = np.where(padded != 0, 1, 0)
        ids_list.append(padded)
        attention_mask_list.append(attention_mask)
    
    if force_device is not None:
        device = torch.device(force_device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)
    if not disable_progress_bar:
        print(f'Gunakan {device} perangkat.')
      
    # dapatkan embedding dalam batch 
    embeddings = []

    for i in tqdm(range(math.ceil(len(ids_list)/batch_size)), disable=disable_progress_bar):
        
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        # <masukkan kode di sini untuk membuat attention_mask_batch 
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size*i:batch_size*(i+1)]).to(device)
           
        with torch.no_grad():            
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)   
        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())
        
    return np.concatenate(embeddings)
```

#### Menjalankan fungsi token dan BERT Text Embedding ke Train & Test sample


```python
%%time

# Perhatian! Menjalankan BERT untuk ribuan teks mungkin memakan waktu lama di CPU, setidaknya beberapa jam
bert_X_train = BERT_text_to_embeddings(df_reviews_train, force_device='cpu')
bert_y_train = df_reviews_train.iloc[:max_sample_size]['pos']
```


```python
print(df_reviews_train['review_norm'].shape)
print(bert_X_train.shape)
print(bert_y_train.shape)
```


```python
%%time

# Perhatian! Menjalankan BERT untuk ribuan teks mungkin memakan waktu lama di CPU, setidaknya beberapa jam
bert_X_test = BERT_text_to_embeddings(df_reviews_test, force_device='cpu')
bert_y_test = df_reviews_test.iloc[:max_sample_size]['pos']
```


```python
print(df_reviews_test['review_norm'].shape)
print(bert_X_test.shape)
print(bert_y_test.shape)
```


```python
# normalisasi feature untuk menghindari nilai error
scaler = MinMaxScaler()
scaler.fit(bert_X_train)

# merubah train dan test dataset menggunakan transform dari scaler
bert_X_train = scaler.transform(bert_X_train)
bert_X_test = scaler.transform(bert_X_test)
```

#### Data Frame hasil Model dengan BERT


```python
# Data Frame hasil Model dengan BERT
log_metrics_bert = {"Models": ["Logistic Regression Classifier", "Gaussian Naive Bayes", "LightGBM Classifier" ,"Stochastic Gradient Classifier" ], "Accuracy_score": [0.0]*4, "F1 Score": [0.0]*4}
m_idx_bert = {"Logistic Regression Classifier":0, "Gaussian Naive Bayes":1, "LightGBM Classifier":2, "Stochastic Gradient Classifier" :3}
```

#### Logistic Regression 


```python
%%time

lr_bert = LogisticRegression(random_state=12345, solver='liblinear')
lr_bert.fit(bert_X_train, bert_y_train)
lr_bert_pred = lr_bert.predict(bert_X_test)
```


```python
acc_lr_bert = accuracy_score(bert_y_test, lr_bert_pred )
f1_sc_lr_bert = f1_score(bert_y_test, lr_bert_pred )
```


```python
log_metrics_bert["Accuracy_score"][m_idx_bert["Logistic Regression Classifier"]] = acc_lr_bert
log_metrics_bert["F1 Score"][m_idx_bert["Logistic Regression Classifier"]] = f1_sc_lr_bert
```


```python
evaluate_model(lr_bert, bert_X_train, bert_y_train, bert_X_test, bert_y_test)
```

#### Gusion naïve Bayes Classifier


```python
%%time
gnb_bert = MultinomialNB()
gnb_bert.fit(bert_X_train, bert_y_train)
gnb_bert_pred = gnb_bert.predict(bert_X_test)
```


```python
acc_gnb_bert = accuracy_score(bert_y_test, gnb_bert_pred)
f1_sc_gnb_bert = f1_score(bert_y_test, gnb_bert_pred)

log_metrics_bert["Accuracy_score"][m_idx_bert["Gaussian Naive Bayes"]] = acc_gnb_bert
log_metrics_bert["F1 Score"][m_idx_bert["Gaussian Naive Bayes"]] = f1_sc_gnb_bert
```


```python
evaluate_model(gnb_bert, bert_X_train, bert_y_train, bert_X_test, bert_y_test)
```

#### LightGBM Classifier


```python
%%time 

lgb_bert = LGBMClassifier(random_state=12345)
lgb_bert.fit(bert_X_train, bert_y_train)
lgb_bert_pred = lgb_bert.predict(bert_X_test)
```


```python
acc_lgb_bert = accuracy_score(bert_y_test, lgb_bert_pred)
f1_sc_lgb_bert = f1_score(bert_y_test, lgb_bert_pred)

log_metrics_bert["Accuracy_score"][m_idx_bert["LightGBM Classifier"]] = acc_lgb_bert
log_metrics_bert["F1 Score"][m_idx_bert["LightGBM Classifier"]] = f1_sc_lgb_bert
```


```python
evaluate_model(lgb_bert, bert_X_train, bert_y_train, bert_X_test, bert_y_test)
```

#### Stochastic Gradient Classifier


```python
%%time

sgd_bert = SGDClassifier(random_state=12345, loss='log')
sgd_bert.fit(bert_X_train, bert_y_train)
sgd_bert_pred = sgd_bert.predict(bert_X_test)
```


```python
acc_sgd_bert = accuracy_score(bert_y_test, sgd_bert_pred)
f1_sc_sgd_bert = f1_score(bert_y_test, sgd_bert_pred)

log_metrics_bert["Accuracy_score"][m_idx_bert["Stochastic Gradient Classifier"]] = acc_sgd_bert
log_metrics_bert["F1 Score"][m_idx_bert["Stochastic Gradient Classifier"]] = f1_sc_sgd_bert
```


```python
evaluate_model(sgd_bert, bert_X_train, bert_y_train, bert_X_test, bert_y_test)
```

## Hasil Model 1, 2 & 3

#### Hasil Model Dengan TF-IDF


```python
pd.DataFrame(log_metrics_tf_idf).sort_values("Accuracy_score", ascending=False).reset_index(drop='index')
```

#### Hasil Model Dengan spaCy


```python
pd.DataFrame(log_metrics_spacy).sort_values("Accuracy_score", ascending=False).reset_index(drop='index')
```

#### Hasil Model Dengan BERT


```python
pd.DataFrame(log_metrics_bert).sort_values("Accuracy_score", ascending=False).reset_index(drop='index')
```

Kesimpulan : 
-----------------------

Pada hasil model menggunakan Natural Languange Preprocessing dari TF-IDF, spaCy dan BERT model `Logistic Regression Classifier` memiliki Accuracy dan F1 Score yang tinggi yakni 1. `Accuracy Score : 0.879796 dan F1 Score 0.879376` pada NLP `TF-IDF` lalu 2. `Accuracy Score : 0.878691 dan F1 Score 0.878753` pada NLP `spaCy` dan 3. `Accuracy Score : 0.814 dan F1 Score 0.826168` pada NLP `BERT`. pada tahp selanjutnya kita akan menggunakan model ML dengan NLP TF-IDF untuk memprediksi ulasan dengan model ML yang telah kita latih. 

## Ulasan Saya


```python
my_reviews = pd.DataFrame([
    'saya tidak begitu menyukainya, bukan jenis film kesukaan saya.', 
    'Membosankan, bahkan saya tidur di tengah-tengah film.', 
    'Filmnya sangat bagus, saya sangat suka',     
    'Bahkan para aktornya terlihat sangat tua dan tidak tertarik dengan filmnya, apakah mereka dibayar untuk bermain film. Sungguh tidak bermutu.', 
    'Saya tidak menyangka filmnya sebagus ini! Para penulis sungguh memperhatikan tiap detailnya', 
    'Film ini memiliki kelebihan dan kekurangan, tetapi saya merasa secara keseluruhan ini adalah film yang layak. Saya mungkin akan menontonnya lagi.', 
    'Beberapa lawakannya sungguh tidak lucu. Tidak ada satu pun lelucon yang berhasil, semua orang bertingkah menyebalkan, bahkan anak-anak pun tidak akan menyukai ini!', 
    'Menayangkan film ini di Netflix adalah langkah yang berani & saya sangat senang bisa menonton episode demi episode dari drama baru yang menarik dan cerdas ini.' 
], columns=['review'])
```


```python
# jangan ragu untuk menghapus ulasan ini dan mencoba model Anda sendiri terhadap ulasan Anda, ini hanyalah sekadar contoh 

my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',    
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])
# clean reviews
my_reviews['review_norm'] = my_reviews['review'].apply(clean_data)# <masukkan logika normalisasi yang sama di sini sebagaimana pada dataset utama>
my_reviews
```


```python
# clean dataset untuk di prediksi
texts = my_reviews['review_norm']
```

### Model 1


```python
# membuat prediksi logistic regression classifier
lr_pred = lr.predict(count_tf_idf.transform(texts))
lr_pred_prob = lr.predict_proba(count_tf_idf.transform(texts))[:, 1]

# ringkasan hasil prediksi
print(f'{sum(lr_pred)} Ulasan diprediksi dengan komentar positive dari {len(lr_pred_prob)} ulasan')
print('='*50)
print()
print('Pos:'+'  '+'Proba:'+'  '+'Actual Ulasan')
for i, review in enumerate(my_reviews['review'].str.slice(0, 100)):
    print(f'  {lr_pred[i]}:   {lr_pred_prob[i]:.2f}:  {review}')
```

### Model 2


```python
# Gusion naïve Bayes Classifier
gnb_pred = gnb.predict(count_tf_idf.transform(texts))
gnb_pred_prob = gnb.predict_proba(count_tf_idf.transform(texts))[:, 1]

# ringkasan hasil prediksi
print(f'{sum(gnb_pred)} Ulasan diprediksi dengan komentar positive dari {len(gnb_pred_prob)} ulasan')
print('='*50)
print()
print('Pos:'+'  '+'Proba:'+'  '+'Actual Ulasan')
for i, review in enumerate(my_reviews['review'].str.slice(0, 100)):
    print(f'  {lr_pred[i]}:   {lr_pred_prob[i]:.2f}:  {review}')
```

### Model 3


```python
# LightGBM Classifier
lgbm_pred = lgb.predict(count_tf_idf.transform(texts))
lgbm_pred_prob = lgb.predict_proba(count_tf_idf.transform(texts))[:, 1]

# ringkasan hasil prediksi
print(f'{sum(lgbm_pred)} Ulasan diprediksi dengan komentar positive dari {len(lgbm_pred_prob)} ulasan')
print('='*50)
print()
print('Pos:'+'  '+'Proba:'+'  '+'Actual Ulasan')
for i, review in enumerate(my_reviews['review'].str.slice(0, 100)):
    print(f'  {lr_pred[i]}:   {lr_pred_prob[i]:.2f}:  {review}')
```

### Model 4


```python
# Stochastic Gradient Classifier
sgd_pred = sgd.predict(count_tf_idf.transform(texts))
sgd_pred_prob = sgd.predict_proba(count_tf_idf.transform(texts))[:, 1]

# ringkasan hasil prediksi
print(f'{sum(sgd_pred)} Ulasan diprediksi dengan komentar positive dari {len(sgd_pred_prob)} ulasan')
print('='*50)
print()
print('Pos:'+'  '+'Proba:'+'  '+'Actual Ulasan')
for i, review in enumerate(my_reviews['review'].str.slice(0, 100)):
    print(f'  {lr_pred[i]}:   {lr_pred_prob[i]:.2f}:  {review}')
```

Kesimpulan :
----
Dari dataset sebanyak 8 ulasan, didapati prediksi dengan 4 ulasan positif dan 4 ulasan negatif. Dari hasil prediksi model yang telah dilatih :
1. Logistik Regresi memprediksi 4 ulasan positif dan 4 ulasan negatif.
2. Gusion naïve Bayes Classifier memprediksi 3 ulasan positif dan 5 ulasan negatif.
3. LightGBM Classifier memprediksi 8 ulasan positif dan 0 ulasan negatif.
4. Stochastic Gradient Classifier memprediksi 6 ulasan positif dan 2 ulasan negatif.

Dari hasil model dengan performa yang terbaik untuk membantu memprediksi ulasan film dari Junky Union adalah Logistik Regresi.

# Daftar Periksa

- [x]  *Notebook* dibuka 
- [ ]  Data teks telah dimuat dan dilakukan pra-pemrosesan untuk vektorisasi 
- [ ]  Data teks telah diubah menjadi vektor 
- [ ]  Model telah terlatih dan diuji 
- [ ]  Ambang batas metrik tercapai 
- [ ]  Semua kode sel tersusun sesuai urutan eksekusinya 
- [ ]  Semua kode sel bisa dieksekusi tanpa *error* 
- [ ]  Terdapat kesimpulan 


```python

```
