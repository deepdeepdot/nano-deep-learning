import urllib.request
import os

def getFilename(link):
    idx = link.rfind("/")
    idx = idx+1 if idx > -1 else 0
    return link[idx:]

def downloadResource(url, folder="."):
    file = getFilename(url)
    path = os.path.join(folder, file)
    print(file, path)
    if os.path.exists(path):
        return # skip already downloaded files (maybe check size > 0?)
    urllib.request.urlretrieve(url, path)

url = "https://news.ycombinator.com/y18.gif"
filename = getFilename(url)

url = "http://s3.amazonaws.com/cadl/celeb-align/000010.jpg"
downloadResource(url)
