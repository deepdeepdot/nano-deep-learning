### II. What's a function?

![The Equation Image](img/e=mc2.png)


#### Running Python scripts

    $ conda activate nanos
    $ ipython
    [1] odds = [n for n in range(10) if n % 2 == 1]
    [2] print("The first odds: ", odds)
    [3] odds. # press tab after the '.' and select 'count'
    [4] !ls # we can execute shell commands
    [5] %save odds.py 1-2
    [6] ?print
    [7]  ?
    [8]] %quickref # 'q' to quit, ctrl+up/down, /pattern, n/p
    [7] quit()
    $ python odds.py
    $ ipython qtconsole
    $ ipython --pylab


### Jupyter Notebooks

* Example<br/>
https://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb

* Tutorial<br/>
https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook


### Running Jupyter
        $ conda activate nanos
        $ conda install jupyter
        $ jupyter notebook . 

- Execute cell: Shift+Return

- How to add the nano kernel to jupyter

        $ conda install nb_conda
        $ conda install ipykernel
        $ ipython kernel install --user --name nanos --display-name "Python (nano)"
        $ jupyter kernelspec list
        $ jupyter kernelspec uninstall unwanted-kernel


### Jupyter Tools

- Post a notebook
    - https://nbviewer.jupyter.org/
    - Share on your github as an .ipynb file

- Jupyterize your github?
    - https://mybinder.org/

- Collaborate on a notebook (google docs)
    - https://colab.research.google.com/


#### Numpy ndarray

    import matplotlib.pyplot as plt

    img = plt.imread("panda-corner.jpg")
    type(img) # prints "numpy.ndarray"

    import numpy as np
    A = np.array([0, 1, 2, 3, 4])
    A.shape
    A[5] = 10 # Array index out of bounds!


#### Numpy basics

    import numpy as np

    A = [[1, 2, 3, 8],
        [2, 0, 3, 9],
        [0, 1, 3, 1]]

    B = [[3, 1], [4, 4], [6, 5], [2, 0]]
    X = np.matmul(A, B)
    X.T

    [1, 2, 3] * 2
    np.array([2, 3, 3]) * 2 + 5

    [1, 2, 3] + [2, 3, 4]
    np.array([1, 2, 3]) + np.array([2, 3, 4])


#### Numpy slicing
    B = np.zeros((3, 4, 5))
    C = B[:,:,0]
    C[0, 0, 0] # Can we do this with a regular array?

    A = [n for n in range(9)]
    A_2D = np.reshape(A, (3, 3))
    A_flat = np.reshape(A_2D, (9))
    A_2D.T

    B = np.arange(27)
    B_3D = B.reshape((3, 3, 3))
    B_flat = B_3D.reshape((3 * 3 * 3))
    B_3D.T

    vals = np.arange(10)
    vals = vals + 10
    idxs = [2, 3, 5]
    vals[idxs]


#### Lambdas

    # Lambda: short inline function
    add10 = lambda a: a + 10
    print(type(add10), add10(23))

    def compute(func, param):
        return func(param)
    
    print("Result: ", compute(add10, 13))

    add = lambda a, b: a + b

    def compute(func, *params): # using *args
        return func(*params)

    print("Result: ", compute(add, 13, 31))


#### Lambdas for a spin with maps

    plus10 = list(map(add10, range(10)))

    additions = list(map(add, [10, 20, 1, 2, 3], [30, 40, 3, 4, 5]))

    mapping = map(add10, range(10))
    for v in mapping: print(v)

    # lambdas are anonymous functions!
    squares = list(map(lambda a: a**2, range(5)))
    odds = list(map(lambda a: 2*a+1, range(5)))

    # Map: http://book.pythontips.com/en/latest/map_filter.html


#### Plotting a function with matplotlib
    import numpy as np
    import pylab as plt

    X = np.linspace(0,5,100)
    Y1 = X + 2*np.random.random(X.shape)
    Y2 = X**2 + np.random.random(X.shape)

    fig, ax = plt.subplots()
    ax.plot(X,Y1,'o')
    ax.plot(X,Y2,'x')
    plt.show()


#### Tutorial on plotting from matplotlib.org

    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()

    # x,y
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

    # x,y, red dots
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.show()


#### Multiple plots on the same graph

    import numpy as np

    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


#### Scatter plotting with different colors and sizes

    data = {'a': np.arange(50),
            'c': np.random.randint(0, 50, 50),
            'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('entry a')
    plt.ylabel('entry b')
    plt.show()


#### Multiple figures

    def f(t):
        return np.exp(-t) * np.cos(2*np.pi*t)

    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

    plt.subplot(212)
    plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
    plt.show()


#### Annotating text

    ax = plt.subplot(111)

    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2*np.pi*t)
    line, = plt.plot(t, s, lw=2)

    plt.annotate('This is the stock market crash', xy=(2, 1), xytext=(3, 1.5),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )

    plt.ylim(-2, 2)
    plt.show()


### Reference
- Colab
    - https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c

- Python Tutorial
    https://docs.python.org/3/tutorial/
    https://www.learnpython.org/


### Matplotlib Reference

- Examples
    - https://matplotlib.org/examples/index.html
    - https://github.com/matplotlib/matplotlib

- Tutorial
    - https://matplotlib.org/tutorials/introductory/pyplot.html
    - https://matplotlib.org/tutorials/index.html
    - https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python


### Animation with Matplotlib

- Animation with Matplotlib
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

- Animation Examples
https://matplotlib.org/examples/animation/index.html

