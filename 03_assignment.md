## Assigment 3: Music generator with Deep Learning

There are two parts for this exercise:
1) Collect a large number of ABC songs into a single text file
2) Generate music using this ABC file as training data

### 1. Crawling for ABC files

The site abcnotation.com has a good search engine to retrieve ABC songs.
The pages we are interested are:
* Search Page
* Search Result Page
* Song Page

The URL for the Search Page: http://abcnotation.com/search

If we search for `"chicken"`, we get the first Search Result Page<br/>
http://abcnotation.com/searchTunes?q=chicken&f=c&o=a&s=0

If we click on next, we get to the second Search Result Page<br/>
http://abcnotation.com/searchTunes?q=chicken&f=c&o=a&s=10

In each Search Result Page, we have links pointing to a Song Page.<br/>
The pattern we see is that each of those links has the text `tune page`

Note: Make sure to create a `"download"` folder.

The algorithm to scrape ABC songs is roughly this:

    For a given search keyword (say "chicken"),
    get the First Search Result Page

    search_result_url = "http://abcnotation.com/searchTunes?q=chicken&f=c&o=a&s=0"
    
    while search_result_url is not None:
        download the search_result_page with that search_result_url
        given the content of this web page, look for all links to a song page
        for each link to a song page:
            download the page with that link
            retrieve the ABC song in that page and save it

        # go to the next link in the search result page (if there's a link)
        search_result_url = "next" link in search_result_page


There's a utility file that you can use for building such scraper
`03_01_get_abc_utils.py` in the `src` folder.

Once all these ABC song text files are saved in the `download` folder, we need to concatenate all these files into a single larger file. Make sure to add few newlines between each ABC song.

Then upload this merged ABC song file to your github repository.

### Some gotchas

#### DDOS Attack?<br>
If you make constant requests without pause to the abcnotation.com server it will reply with some 400 permission error. You might be contributing to some DDOS attack. Hence, you want to wait for 5 seconds between each web request.

    import time
    time.sleep(5) # wait for 5 seconds

#### Cleanup the ABC data
I noticed some strange lines, which I deleted whenever I see them

    I:abc-charset utf-8

I also removed the ABC songs that contained a header but no notes.
Training with empty notes will train the neural network to create empty songs. This is not something we want to encourage :)

Rarely, I see HTML content instead of an ABC song. So I delete this too.

#### Massive amount of data?
If the quality of the resulting music is not very musical (too random?), it might be because the neural network didn't have enough data. What's `enough` data? I tried with the `chinese` keyword and got about 100 songs. The quality wasn't good, it didn't sound like a Chinese folk tune. The Irish ABC sample contains 816 ABC songs and the sound was pretty good. Therefore, if you don't have enough for a certain keyword, try few different keyword searchs and concatenate all the ABC song files into a single larger one that has mre than 1000 ABC songs.


### 2. Music generation with Deep Learning

Prerequisites: make sure to upload your ABC song file in github.

As we have seen in class, these are the steps we followed:
1) Visit the jupyter notebook
https://github.com/aamini/introtodeeplearning_labs/blob/master/lab1/Part2_music_generation_solution.ipynb

2) Open the link `"Run in Google Colab"`, which willl open the Jupyter notebook on Google Colab using the GPU runtime. You'll need to sign in with your gmail account to access this page.

3) Execute each cell in order (from top to bottom)
Click on the play arrow when you hover over the `[ ]` on the left side.

    Note: on the second cell, you need to replace `'1.13.0'` with `'1.13.1'` (as of March 15, 2019)

4) We need to replace the *`path_to_file`* line with the github raw link to your ABC text file.

        path_to_file = tf.keras.utils.get_file('irish.abc', 'https://raw.githubusercontent.com/aamini/introtodeeplearning_labs/2019/lab1/data/irish.abc')

    For instance, if you visit:
https://raw.githubusercontent.com/aamini/introtodeeplearning_labs/2019/lab1/data/irish.abc

    You will see a collection of Irish folk songs. We want to replace this URL with the one that has *`your`* ABC file.

5) Once you reach to the last cell, copy/paste the resulting ABC songs into a text file, which you'll upload to your github. You can export this abc to a midi file using `EasyABC`. Upload the midi to your github.


## Slope approximation

We have seen in class the definition of a slope given two points.
When trying to find the minimum of a function, we are looking for the place where the slope is zero.

    Y = lambda x: x**2 - 3*x + 10
    plt.plot(X, Y(X))

    Pick 2 points, compute the slope
    p1 = (3, Y(3)) # 10?
    p2 = (4, Y(4)) # 14?
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    # slope = 4    # quite steep!
    plt.plot(X, X*slope)
    plt.plot(X, X*slope-2.4)

Then we used a step=-1 (left), since slope=4 (positive)<br>
and used step=-2 (left), since the slope=2 (still positive)<br>
and used step=+1 (right), since the slope=-2 (negative)<br>

We followed the following algorithm:

    Loop
        Compute the slope at current position
        Is the slope == 0, break
        Is the slope positive? go to the left
        Is the slope negative? go to the right

The exercise is to implement this algorithm in a function

    def findMinimum(Y):
        # TODO: implement findMinimum()
        # Find out p1, p2 with slope=0 and small distance between p1 and p2
        # Count how many iterations we used to find out the slope = 0 (or close to zero)
        p1 = None
        p2 = None
        iterations = None
        return (p1, p2, iterations)

    Y = lambda x: x**2 - 3*x + 10
    p1, p2, iterations = findMinimum(Y)
    print(p1, p2, iterations)

    Try different lambda functions for Y

