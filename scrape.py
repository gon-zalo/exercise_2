# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import requests
from bs4 import BeautifulSoup
import re

# SCRAPING NAMES LIST --------------------------------------------------------

names_list = []
urls = ['https://www.znaczenie-imion.net/europejskie/polskie-imiona.html', 
        'https://www.znaczenie-imion.net/europejskie/polskie-imiona.html/2']

# to avoid getting flagged
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

page = 1 # just to print while the code runs
for url in urls:
    print(f'Fetching names... Page number: {page}')

    # gets the html of the url
    response = requests.get(url, headers=headers) 
    soup = BeautifulSoup(response.text, 'html.parser')

    # this is where the names are stored in the website
    div_tag = soup.find('div', class_='thecontent')

    # the names are all in a li tag inside the previous div tag
    li_tags = div_tag.find_all('li')

    for name in li_tags: # li_tags is a list, we loop through it and append each name in lowercase to names_list
        names_list.append(name.get_text().lower())

    page += 1

print(names_list)
# saving names_list in a .txt file
with open ('names.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(names_list))


# SCRAPING OLD NAMES CODE -----------------------------------------------------------

old_names = []
url = 'https://pl.wikisource.org/wiki/Encyklopedia_staropolska/Imiona_staro-polskie'

# to avoid getting flagged
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# gets the html of the url
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# all the names are inside an i tag (italics), the rest of the names are other spellings or colloquializations of the name in italics
names = soup.find_all('i')

for name in names:
    old_names.append(name.get_text().lower())

cleaned_names = []
# processing the data to remove unwanted characters
for name in old_names:
    split_name = re.split(r'[;,.]+', name)  # split by ;, or .
    cleaned_names.extend(split_name) #  makes it so that the split names are added as individual elements in the list.
    cleaned_names = [name.strip() for name in cleaned_names] # removes empty spaces in strings
    cleaned_names = [name for name in cleaned_names if name] # removes empty strings

print(cleaned_names)
# saving names_list in a .txt file
with open ('old_names.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned_names))