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


def getABCLinksAndNextSearchPageURL(searchResultHtml):
    abcLinks = []
    nextPage = None
    
    soup = BeautifulSoup(searchResultHtml, 'html.parser')
    for link in soup.find_all('a'):
        if link.get_text().find("tune page") == 0:
            abcLinks.append(link.get('href'))
        if link.get_text() == "next":
            nextPage = link.get('href')
    
    base = "http://abcnotation.com"
    links = [base + link for link in abcLinks]
    if nextPage is not None:
        nextPage = base + nextPage
    return (links, nextPage)

# Sample program using getABCLinksAndNextSearchPageURL(searchResultHtml)
url = "http://abcnotation.com/searchTunes?q=chicken&f=c&o=a&s=0"

searchResultFilename = "search_result_chicken_00.html"
downloadResource(url, "download", searchResultFilename)
searchResultHtml = open(f"download/{searchResultFilename}", 'r').read() 

links, nextSearchResultPageURL = getABCLinksAndNextSearchPageURL(searchResultHtml)
print("Next Search Result URL:", nextSearchResultPageURL)
print("Song Links: ", links)


def getABCSong(songHtml):
    soup = BeautifulSoup(songHtml, 'html.parser')
    textarea = soup.find("textarea")
    song = textarea.contents[0].strip()
    return song

# Sample program using getABCSong(songHtml)
url = "/tunePage?a=ifdo.ca/~seymour/runabc/esac/HAN2/0495"
downloadResource("http://abcnotation.com" + url, "download", "0495.html")
songHtml = open('download/0495.html', 'r').read()
song = getABCSong(songHtml)

print(song)

