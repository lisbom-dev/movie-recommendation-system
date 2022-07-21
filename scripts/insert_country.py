import numpy as np
import random
import csv

unique_user_ids = np.array(range(944)).astype(str)

user_country = {}
country_list = ['Macedonia',
                'Anguilla',
                'American Samoa',
                'Bhutan',
                'Tuvalu',
                'Azerbaijan',
                'Saudi Arabia',
                'Argentina',
                'Brazil',
                'Bolivia',
                'Saint Helena',
                'Montenegro',
                'Uruguay',
                'China',
                'Italy']

for id in unique_user_ids:
    user_country[id] = random.choice(country_list)


newcsv = []
with open('data/TFRS-movie-ranking/ratings.csv') as file:
    mycsv = csv.DictReader(file)

    for row in mycsv:
        row['country'] = user_country[str(row['userId'])]
        newcsv.append(row)


with open('data/TFRS-movie-ranking/ratings2.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=['userId','userAge','country','movieId','movieGenres','rating','timestamp'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(newcsv)
