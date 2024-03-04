
import csv
from math import sqrt, pow, exp


#converts tab separated .txt file to .csv
def text_to_csv(filename):

    with open(f"./YelpZip/{filename}.txt", encoding='cp850') as f:
        text = f.read().split('\n')

    with open('{filename}.csv', mode='w', newline='', encoding='cp850') as content_file:
        content_writer = csv.writer(content_file, delimiter=',', quotechar='"',  quoting=csv.QUOTE_MINIMAL)

        for line in text:
            content_writer.writerow(line.split('\t'))