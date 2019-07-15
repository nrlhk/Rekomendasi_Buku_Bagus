import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    'books.csv'
)
# print(df.head(2))
# print(df.columns.values)
# ['book_id' 'goodreads_book_id' 'best_book_id' 'work_id' 'books_count'
#  'isbn' 'isbn13' 'authors' 'original_publication_year' 'original_title'
#  'title' 'language_code' 'average_rating' 'ratings_count'
#  'work_ratings_count' 'work_text_reviews_count' 'ratings_1' 'ratings_2'
#  'ratings_3' 'ratings_4' 'ratings_5' 'image_url' 'small_image_url']
# print(df.shape) # (10000, 23)

df = df[['original_title', 'authors', 'average_rating']]
# print(df.head())
df = df.dropna()
# print(df.isnull().sum())

newReview = [
    {'nama': 'Andi', 'judul': 'The Hunger Games', 'rating': 5},
    {'nama': 'Andi', 'judul': 'Catching Fire', 'rating': 5},
    {'nama': 'Andi', 'judul': 'Mockingjay', 'rating': 4},
    {'nama': 'Andi', 'judul': 'The Hobbit or There and Back Again', 'rating': 4},
    {'nama': 'Andi', 'judul': 'Animal Farm: A Fairy Story', 'rating': 1},
    {'nama': 'Budi', 'judul': 'Harry Potter and the Chamber of Secrets', 'rating': 5},
    {'nama': 'Budi', 'judul': 'Harry Potter and the Philosopher\'s Stone', 'rating': 5},
    {'nama': 'Budi', 'judul': 'Harry Potter and the Prisoner of Azkaban', 'rating': 5},
    {'nama': 'Ciko', 'judul': 'The Brightest Star in the Sky', 'rating': 2},
    {'nama': 'Ciko', 'judul': 'The Last Seven Months of Anne Frank', 'rating': 1 },
    {'nama': 'Ciko', 'judul': 'The Venetian Betrayal', 'rating': 2},
    {'nama': 'Ciko', 'judul': 'Robots and Empire', 'rating': 5},
    {'nama': 'Dedi', 'judul': 'Nine Parts of Desire: The Hidden World of Islamic Women', 'rating': 4},
    {'nama': 'Dedi', 'judul': '"A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam"', 'rating': 5},
    {'nama': 'Dedi', 'judul': '"No god but God: The Origins, Evolution, and Future of Islam"', 'rating': 4},
    {'nama': 'Dedi', 'judul': 'Hunter × Hunter #1', 'rating': 1},
    {'nama': 'Dedi', 'judul': 'Peter Pan', 'rating': 2},
    {'nama': 'Ello', 'judul': 'Being Mortal: Medicine and What Matters in the End', 'rating': 2},
    {'nama': 'Ello', 'judul': 'Georges Marvellous Medicine', 'rating': 2},
    {'nama': 'Ello', 'judul': 'Doctor Sleep', 'rating': 4},
    {'nama': 'Ello', 'judul': 'The Story of Doctor Dolittle', 'rating': 5},
    {'nama': 'Ello', 'judul': 'Bridget Joness Diary', 'rating': 5},
]
dfProduk = pd.DataFrame(newReview)
dataAndi = dfProduk[dfProduk['nama'] == 'Andi']
rekAndi = dataAndi[dataAndi['rating'] == dataAndi['rating'].max()]['judul'].iloc[0]

dataBudi = dfProduk[dfProduk['nama'] == 'Budi']
rekBudi = dataBudi[dataBudi['rating'] == dataBudi['rating'].max()]['judul'].iloc[0]

dataCiko = dfProduk[dfProduk['nama'] == 'Ciko']
rekCiko = dataCiko[dataCiko['rating'] == dataCiko['rating'].max()]['judul'].iloc[0]

dataDedi = dfProduk[dfProduk['nama'] == 'Dedi']
rekDedi = dataDedi[dataDedi['rating'] == dataDedi['rating'].max()]['judul'].iloc[0]

dataEllo = dfProduk[dfProduk['nama'] == 'Ello']
rekEllo = dataEllo[dataEllo['rating'] == dataEllo['rating'].max()]['judul'].iloc[0]

# print(rekAndi)
# print(rekBudi)
# print(rekCiko)
print(rekDedi)
# print(rekEllo)

# Count authors
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer = lambda i: i.split('@'),
    analyzer= 'word'
)

matrixauthors = model.fit_transform(df['authors'])
# print(matrixauthors)

author = model.get_feature_names()
# print(author)
jumlahGenre = len(author)
# print(jumlahGenre)      # 4664
eventauthor = matrixauthors.toarray()
# print(eventauthor[0])

# Cosinus Similarity
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixauthors)
# print(score)

#test model
Andi = rekAndi
# 'Catching Fire', 'Mockingjay', 'The Hobbit or There and Back Again', 'Animal Farm: A Fairy Story']
indexandi = df[df['original_title'] == Andi].index.values[0]
# print(indexandi)

allauthorsA = list(enumerate(score[indexandi]))
# print(allauthorsA)

authorSama = sorted(
    allauthorsA,
    key = lambda i: i[1],
    reverse = True
)

authorSama70up = []
for i in authorSama:
    if i[1] >= 0.5:
        authorSama70up.append(i)
# print(authorSama70up)

import random
rekomendasi = random.choices(authorSama70up, k = 5)
# print(rekomendasi)
print('Buku bagus untuk Andi:')
for i in rekomendasi:
    print(
        df.iloc[i[0]]['original_title'],
        )

#test model
Budi = rekBudi
indexBudi = df[df['original_title'] == Budi].index.values[0]

allauthorsB = list(enumerate(score[indexBudi]))
# print(allauthors)

authorSamaB = sorted(
    allauthorsB,
    key = lambda i: i[1],
    reverse = True
)

authorSama70upB = []
for i in authorSamaB:
    if i[1] >= 0.5:
        authorSama70upB.append(i)
print(authorSama70upB)

rekomendasiB = random.choices(authorSama70upB, k = 5)
# print(rekomendasi)
print('Buku bagus untuk Budi:')
for i in rekomendasiB:
    print(
        df.iloc[i[0]]['original_title'],
        )

#test model
Ciko = rekCiko
indexCiko = df[df['original_title'] == Ciko].index.values[0]

allauthorsC = list(enumerate(score[indexCiko]))

authorSamaC = sorted(
    allauthorsC,
    key = lambda i: i[1],
    reverse = True
)

authorSama70upC = []
for i in authorSamaC:
    if i[1] >= 0.5:
        authorSama70upC.append(i)
# print(authorSama70upC)

rekomendasiC = random.choices(authorSama70upC, k = 5)
# print(rekomendasi)
print('Buku bagus untuk Ciko:')
for i in rekomendasiC:
    print(
        df.iloc[i[0]]['original_title'],
        )

#test model
Dedi = "A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam"
indexDedi = df[df['original_title'] == Dedi].index.values[0]

allauthorsD = list(enumerate(score[indexDedi]))

authorSamaD = sorted(
    allauthorsD,
    key = lambda i: i[1],
    reverse = True
)

authorSama70upD = []
for i in authorSamaD:
    if i[1] >= 0.5:
        authorSama70upD.append(i)
# print(authorSama70upD)

rekomendasiD = random.choices(authorSama70upD, k = 5)
# print(rekomendasi)
print('Buku bagus untuk Dedi:')
for i in rekomendasiD:
    print(
        df.iloc[i[0]]['original_title'],
        )

#test model
Ello = rekEllo
indexEllo = df[df['original_title'] == Ello].index.values[0]

allauthorsE = list(enumerate(score[indexEllo]))

authorSamaE = sorted(
    allauthorsE,
    key = lambda i: i[1],
    reverse = True
)

authorSama70upE = []
for i in authorSamaE:
    if i[1] >= 0.5:
        authorSama70upE.append(i)
rekomendasiE = random.choices(authorSama70upE, k = 5)

print('Buku bagus untuk Ello:')
for i in rekomendasiE:
    print(
        df.iloc[i[0]]['original_title'],
        )
