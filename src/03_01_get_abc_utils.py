# conda install -c anaconda beautifulsoup4
from bs4 import BeautifulSoup
import urllib.request
import os

def getFilename(link):
    idx = link.rfind("/")
    idx = idx+1 if idx > -1 else 0
    return link[idx:]

def downloadResource(url, folder='.', name=None):
    file = name or getFilename(url)
    path = os.path.join(folder, file)
    if os.path.exists(path):
        return # skip already downloaded files (maybe check size > 0?)
    urllib.request.urlretrieve(url, path)


# Sample demos
url = "https://news.ycombinator.com/y18.gif"
filename = getFilename(url)

url = "http://s3.amazonaws.com/cadl/celeb-align/000010.jpg"
downloadResource(url)


def getABCLinksAndNextSearchPageURL(html_doc):
    abcLinks = []
    nextPage = None
    
    soup = BeautifulSoup(html_doc, 'html.parser')
    for link in soup.find_all('a'):
        if link.get_text().find("tune page") == 0:
            abcLinks.append(link.get('href'))
        if link.get_text() == "next":
            nextPage = link.get('href')
    
    base = "http://abcnotation.com"
    links = [base + link for link in abcLinks]
    return (links, base + nextPage)


url = "http://abcnotation.com/searchTunes?q=china&f=c&o=a&s=0"
html_filename = "search_result_china_00.html"
downloadResource(url, "download", html_filename)
html_doc = open(f"download/{html_filename}", 'r').read() 

links, nextSearchResultPageURL = getABCLinksAndNextSearchPageURL(html_doc)

print("Next Search Result URL:", nextSearchResultPageURL)
print("Song Links: ", links)