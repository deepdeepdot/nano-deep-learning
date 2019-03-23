### III. Music and Slope


### Topics

* Music generation with Deep Learning
* ABC files
* Scraping web resources
* Slope of a function


### Irish Tunes with Deep Learning

- Visit
https://github.com/aamini/introtodeeplearning_labs/blob/master/lab1/Part2_music_generation_solution.ipynb

- Click on "Run in Google Colab"
- Execute each cell


### Make your own music generator

- Workflow: ABC -> MIDI -> WAVE
- We need to crawl for ABC files on the Internet.
- Merged all these ABC files into a single text file.
- Upload this text file to github
- Replace `path_to_file` with your raw github link
- Retrieve multiple produced ABC text and save these to your github.
- Extra: convert ABC to midi using `abc2midi`, convert midi to wave with `timidity`


### ABC 

* EasyABC<br/>
https://www.nilsliberg.se/ksp/easyabc/

* Examples<br/>
http://abcnotation.com/examples


#### ABC Sample #1

    X:1
    T:Notes
    M:C
    L:1/4
    K:C
    C, D, E, F,|G, A, B, C|D E F G|A B c d|e f g a|b c' d' e'|f' g' a' b'|]


### Data sets
    * http://deeplearning.net/datasets/
    * https://www.kaggle.com/datasets

    - CIFAR10
    - MNIST
    - ImageNet
    - MovieLens
    - Celebs
    - MSCOCO

    Celebs, 200,000 images of celebrities
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


#### How to scrape resources from the web

* Machine learning depends on large amounts of data

* For music, let's google for ABC music files
* How we can we scrape for ABC music files? (discuss algorithm)

    For Chinese tunes:
    Hint: http://abcnotation.com/searchTunes?q=china

* How about for midi files?


#### Get the filename from a URL

    def getFilename(link):
        idx = link.rfind("/")
        idx = idx+1 if idx > -1 else 0
        return link[idx:]

    url = "https://news.ycombinator.com/y18.gif"
    filename = getFilename(url)


#### Download a resource from the web

    import urllib.request
    import os

    def downloadResource(url, folder="."):
        file = getFilename(url)
        path = os.path.join(folder, file)
        if os.path.exists(path):
            return # skip already downloaded files (maybe check size > 0?)
        urllib.request.urlretrieve(url, path)


#### Download a web resource

    # Download an image
    url = "https://news.ycombinator.com/y18.gif
    downloadResource(url, "download")

    # Download the MP3
    url = "/getResource/resources/media/ai-erwa.mp3"
    downloadResource("http://abcnotation.com" + url, "download", "chinese.mp3")

    # Download an ABC song page
    url = "http://abcnotation.com/getResource/resources/media/ai-erwa.mp3?a=ifdo.ca/~seymour/runabc/esac/HAN2/0495/"
    downloadResource(url, "download")

    url = "/tunePage?a=ifdo.ca/~seymour/runabc/esac/HAN2/0495"
    downloadResource("http://abcnotation.com" + url, "download")

    # I'd rather save as HTML: "0495.html"
    downloadResource("http://abcnotation.com" + url, "download", "0495.html")


#### Add a default 'file' parameter

    def downloadResource(url, folder='.', name=None):
        file = name or getFilename(url)
        path = os.path.join(folder, file)
        if os.path.exists(path):
            return # skip already downloaded files (maybe check size > 0?)
        urllib.request.urlretrieve(url, path)

    # Download an ABC song page
    url = "/tunePage?a=ifdo.ca/~seymour/runabc/esac/HAN2/0495"
    downloadResource("http://abcnotation.com" + url, "download", "0495.html")


#### Download the ABC song from the URL

    content = open("download/0495.html", 'r').read()
    abcStartTag = '<textarea cols="62" rows="13" readonly="readonly">'
    start = content.find(abcStartTag) + len(abcStartTag)
    end = content.find("</textarea>", start)

    abcSong = content[start:end].strip()
    print(abcSong, file=open('0495.abc', 'w'))


#### Skip saving the intermediate HTML page

    url = "/tunePage?a=ifdo.ca/~seymour/runabc/esac/HAN2/0495"
    req = urllib.request.urlopen("http://abcnotation.com" + url)
    content = req.read()

    abcStartTag = '<textarea cols="62" rows="13" readonly="readonly">'
    start = content.find(abcStartTag) + len(abcStartTag)
    end = content.find("</textarea>", start)

    abcSong = content[start:end]
    print(abcSong, file=open('0495.abc', 'w'))


#### Web scraper for ABC files for Chinese tunes?

    Go to the search result page (starting at s=0)
    url = "http://abcnotation.com/searchTunes?q=chinese&f=c&o=a&s=0"

    for each search_result_page
        find all links to song pages
        for each link in song pages:
            get the content of the song page
            retrieve the ABC song and save it

    The HTML structure looks so weird... 
    Maybe it's time for some beautiful soup!!


#### Beautiful Soup in Python

    # conda install -c anaconda beautifulsoup4
    from bs4 import BeautifulSoup

    # Song page
    html_doc = open('download/0495.html', 'r').read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    textarea = soup.find("textarea")
    song = textarea.contents[0].strip()
    links = soup.find_all('a')

    for link in links:
        print(link.get_text()) 

    # See: https://www.crummy.com/software/BeautifulSoup/bs4/doc/


#### Get all the abc links from the search result

    # Search result page
    url = "http://abcnotation.com/searchTunes?q=china&f=c&o=a&s=0"
    html_filename = "search_result_china_00.html"
    downloadResource(url, "download", html_filename)
    html_doc = open(f"download/{html_filename}", 'r').read() 
    soup = BeautifulSoup(html_doc, 'html.parser')

    abcLinks = []
    for link in soup.find_all('a'):
        if link.get_text().find("tune page") == 0:
            abcLinks.append(link.get('href'))
        if link.get_text() == "next":
            nextPage = link


#### Read and Write a file in Python

    filename = "myfile.txt"
    file = open(filename, "w") # "a" for append

    file.write("Roses are red")
    file.write("Violets are blue")
    file.write("Sugar is sweet")

    file.close()

    file = open(filename, "r") # "r": read
    content = file.read()
    print(content)


### Reference

* For the serious scrapper
### https://scrapy.org/

- https://github.com/scrapy/scrapy


#### Slope - rate of change

### m = (y2 - y1) / (x2 - x1)

- It's like a ladder
* m=1: for every step in X, we move one step in Y
* m=2: for every step in X, we move 2 steps in Y
* m=10: for every step in X, 10 steps in Y
* m=0.5: for every step in X, half a step in Y
* m=-1: for every step in X, one step back in Y


#### Slope

    %pylab
    X = np.linspace(-5,5,100)
    plt.plot(X, X**2-3*X+10)

    # Let's plot the slope for (4,14) - (3,10)
    # slope = (y2 - y1) / (x2 - x1) = (14-10) / (4-3) = 4
    plt.plot(X, 4*X) # Not quite there as the tangent

    plt.plot(X, 4*X-3) # Quite close!
    plt.plot(X, 4*X-2.4) # Quite close!


#### How do we find the minimum of a function?
    - By approximation. How? How many steps?

    Y = lambda x: x**2 - 3*x + 10
    plt.plot(X, Y(X))

    Pick 2 points, compute the slope
    p1 = (3, Y(3)) # 10?
    p2 = (4, Y(4)) # 14?
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # slope = 4    # quite steep!
    plt.plot(X, X*slope)
    plt.plot(X, X*slope-2.4)
    plt.plot(X, X)


#### Let's go to the right?

    # How do we find out the zero slope?
    # Well, the slope is 4, very steep and positive.
    # For a "convex" function, we want to go opposite the slope direction

    step = -1
    p1 = (2, Y(2))
    p2 = (3, Y(3))
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # slope = 2.0 # The slope has decreased! Keep going!
    plt.plot(X, slope*X)
    plt.plot(X, slope*X+4)


#### Let's keep searching for the minimum
    step = -2
    p1 = (0, Y(0))
    p2 = (1, Y(1))
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # slope = -2  # Ooops. We are negative!
    plt.plot(X, X*slope) 

    step = +1
    p1 = (1, Y(1))
    p2 = (2, Y(2))
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # slope = 0 
    plt.plot(X, slope*X)
    plt.plot(X, slope*X+8)


#### Are we there?
    # But there are many points with slope = 0
    # Let's get a "better" approximation to Min(x,Y(x))

    step=.49, .51
    p1 = (1.49, Y(1.49))
    p2 = (1.51, Y(1.51))
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # slope = 0 # Still zero, but this point is closer to the truth
    # Guessing that the minimum is at (1.5, Y(1.5))?

    Algorithm
    Loop
        Compute the slope at current position
        Is the slope == 0, break
        Is the slope positive? go to the left
        Is the slope negative? go to the right


#### Slope competition

    Who will implement the "fastest" search 
    for the minimum given a lambda as an input. 
    "Fastest" is defined as the minimum number of 
    iterations to get the minimum.

    minimum, steps = find_minimum(some_lambda)

    (1.5, Y(1.5)), 10  # 10 steps to get estimate the minimum 1.5, Y(1.5))

    The challenge is how to determine the "step" to get to the next 2 points to get the slope.


#### Music with Magenta.js

    Hello Magenta
    https://medium.com/@oluwafunmi.ojo/getting-started-with-magenta-js-e7ffbcb64c21
    https://hello-magenta.glitch.me/

    Melody Mixer
    https://experiments.withgoogle.com/ai/melody-mixer/view/
    https://github.com/googlecreativelab/melody-mixer

