# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import requests
from bs4 import BeautifulSoup
import re

# old SCRAPING CODE ---------------------------------------------------------------------

# names_list = [] # empty list to append the names to
# for page in range(1, 14): # there are 13 pages in the website

#     url = f'https://imiennik.net/imiona-meskie.html?cp_page={page}'
#     print(f'Fetching names... Current page: {page}')
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser') # gets the html of the url
#     table = soup.find('table', class_='tabb') # gets the table where the names are stored, this is needed so that the code does not fetch other names in the website

#     names = table.find_all('a', attrs={'boy'}) # getting the names out of the table, they are in an a tag 
#     for name in names: # names is a list, for loop to iterate over the list and append the names to the list above
#         names_list.append(name.get_text())

# for name in names_list: # 'imiona męskie' means 'male names', it is under an a tag too in the website and it gets scraped, this is a simple quick fix
#     if name == 'Imiona męskie': 
#         names_list.remove(name)

# # saving names_list in a .txt file
# with open ('names.txt', 'a', encoding='utf-8') as f:
#     for name in names_list:
#         print(name.lower(), file=f)
#   f.write('\n'.join(names_list))


# SCRAPING NAMES LIST --------------------------------------------------------

# names_list = []
# urls = ['https://www.znaczenie-imion.net/europejskie/polskie-imiona.html', 
#         'https://www.znaczenie-imion.net/europejskie/polskie-imiona.html/2']

# # to avoid getting flagged
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# page = 1 # just to print it while the code runs
# for url in urls:
#     print(f'Fetching names... Page number: {page}')

#     # gets the html of the url
#     response = requests.get(url, headers=headers) 
#     soup = BeautifulSoup(response.text, 'html.parser')
#     # this is where the names are stored
#     div_tag = soup.find('div', class_='thecontent')
#     # the names are all in a li tag
#     li_tags = div_tag.find_all('li')

#     for name in li_tags: # li_tags is a list, we loop through it and append each name in lowercase to names_list
#         names_list.append(name.get_text().lower())

#     page += 1

# # saving names_list in a .txt file
# with open ('names.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(names_list))


# SCRAPING OLD NAMES CODE -----------------------------------------------------------

# old_names = []
# url = 'https://pl.wikisource.org/wiki/Encyklopedia_staropolska/Imiona_staro-polskie'

# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')

# names = soup.find_all('i')

# for name in names:
#     old_names.append(name.get_text().lower())

# cleaned_names = []
# for name in old_names:
#     split_name = re.split(r'[;,.]+', name)  # Split by ;, or .
#     cleaned_names.extend(split_name) #  Ensures that names separated by ; are added as individual elements in the list.
#     cleaned_names = [name.strip() for name in cleaned_names] # removes empty spaces in strings
#     cleaned_names = [name for name in cleaned_names if name] # removes empty strings

# print(cleaned_names)
# saving names_list in a .txt file
# with open ('old_names2.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(cleaned_names))