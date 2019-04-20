### V. Text and Language


### Topics

* Calculator
* Recursion
* Trees
* Tensorflow as Calculator
* Karpathy's RNN
* Music RNN
* Magenta.js


### Life of a Calculator

- Low-level: Little Man Computer?
  * https://en.wikipedia.org/wiki/Little_man_computer
  * https://hacks.mozilla.org/2017/02/a-crash-course-in-assembly/

- High-level:
  * Parser: "4 / 2 + 3 * 4" -> tree
  * Eval: tree -> number


### What's recursion?

    Who is Remus?
    Remus is a the brother of Romulus

    And who is Romulus?
    Romulus is the brother of Remus


### L-Systems (Lindenmayer)

* https://en.wikipedia.org/wiki/L-system
  - Parallel rewriting system
  - Type of formal grammar

* [P5.js L-Systems](https://p5js.org/examples/simulate-l-systems.html)
* [P5.js Penrose Tiles](https://p5js.org/examples/simulate-penrose-tiles.html)
* [Hilbert Curve - wikipedia](https://en.wikipedia.org/wiki/Hilbert_curve)


### Rules of Conway's Game of Life

1. Any live cell with fewer than two live neighbours dies,
as if by underpopulation.
2. Any live cell with two or three live neighbours lives on to the next generation.
3. Any live cell with more than three live neighbours dies,
as if by overpopulation.
4. Any dead cell with exactly three live neighbours becomes a live cell,
as if by reproduction.


### Conway's Game of Life
* https://www.youtube.com/watch?v=C2vgICfQawE

* http://xahlee.info/math/recursive_game_of_life.html
* https://nullprogram.com/blog/2014/06/10/

* Mandelbrots Fractals
http://blog.recursiveprocess.com/2014/04/05/mandelbrot-fractal-v2/


#### Recursive definitions

    Fibonacci
    fib(0) = 0
    fib(1) = 1
    fib(n) = fib(n-1) + fib(n-2) # for n > 1

    Python challenge: implement fibonacci

    Factorial
    fact(0) = 1
    fact(1) = 1
    fact(n) = n * fact(n-1) # for n > 1

* Python challenge: implement factorial and fibonacci


### Recursion visually

* https://blog.penjee.com/how-recursion-works-in-7-gifs/

munificent/generate.c
* https://gist.github.com/munificent/b1bcd969063da3e6c298be070a22b604
A random dungeon generator that fits on a business card 


#### Recursive definitions: Expressions

    # Rules
    expression = [number]
    expression = [operation, expression, expression]

    num_expression = [6]

    # Expression for: "4 / 2 + 3 * 4"
    computational_expression = [
      '+',
      ['/', [4], [2]],
      ['*', [3], [4]]
    ]

* Lisp: a functional language based on lists
* https://twobithistory.org/2018/10/14/lisp.html


####  Print the expression recursively

    def print_expression(expression):
      if (len(expression) == 1):
        print(expression[0])
      else:
        print(expression[0])
        print_expression(expression[1])
        print_expression(expression[2])

    print_expression(num_expression)
    print_expression(computational_expression)

    Python challenge: how can we eval() the expression recursively?


####  Compute the expression

    operations = {
      '+': lambda a, b: a + b,
      '-': lambda a, b: a - b,
      '*': lambda a, b: a * b,
      '/': lambda a, b: a / b
    }

    def eval_expression(expression):
      if (len(expression) == 1):
        return expression[0] # must be a number, right?
      else:
        operand = operations[ expression[0] ]
        left = eval_expression( expression[1] )
        right = eval_expression( expression[2] )
        return operand(left, right)

    print("Total: ", eval_expression(computational_expression))


#### Tree: Node class

    Class Node:
      data
      right (pointer to Node)
      left (pointer to a Node)
      isLeaf(): right and left are None

    def eval(<data>, left=a=node, right=a-node):
      return 1


#### Tree: let's plant one

    def node(data=None, left=None, right=None):
      return {
        "data": data,
        "left": left,
        "right": right
      }

    root = node('+',
      left=node('/',
        left=node('4'),
        right=node('2')
      ),
      right=node('*',
        left=node('3'),
        right=node('4')
      )
    )


#### let's traverse!

    def visit(node):
      if node != None:
        visit(node["left"])
        print(node["data"]) # in-order traversal, pre/post?
        visit(node["right"])
      
    visit(root)
    # What if we "print" before visiting right and left? Pre-order
    # What if we "print" after visiting right and left? Post-order


### Node class

    # Expression for: "4 / 2 + 3 * 4"
    root = Node(operations['+'],
      left=Node(operations['/'], left=Node(4), right=Node(2)),
      right=Node(operations['*'], left=Node(3), right=Node(4))
    )
    total = root.eval(root)

    class Node:
      def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

      def eval(self, node):
        return []

Implement 'eval()'


#### Node class

    class Node:
      def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

      def eval(self, node):
        if type(node.data) == int or type(node.data) == float:
          return node.data

        value_left = self.eval(node.left)
        value_right = self.eval(node.right)
        return node.data( value_left, value_right )


### Parser

* What's missing in our calculator?
  - It's hardcoded for the expression: "4/2+3*4"
  - No support for parenthensis, unary operators (negative)
  - How can we make it dynamic? How do we support variables?

* Yacc and Lex for Python
  - https://www.dabeaz.com/ply/
  - https://www.dabeaz.com/ply/ply.html
  - https://github.com/dabeaz/ply


### Tree for source code?

* Calculator: input = math expression
* Interpreter: input = computer program source code
    - operations: for-loop, if, switch, expressions
* Reference
  - http://openbookproject.net/thinkcs/python/english3e/trees.html
  - https://pypi.org/project/binarytree/


#### Tensorflow is a computational graph
    # $ conda activate nanos
    # $ conda install tensorflow # or tensorflow-gpu
    import tensorflow as tf

    sess = tf.Session() # Create a session

    # z = "4 / 2 + 3 * 4"
    z = tf.add(tf.divide(4.0, 2.0), tf.multiply(3.0, 4.0))
    z = tf.add(tf.divide(tf.constant(4.0), tf.constant(2.0)),
                tf.multiply(tf.constant(3.0), tf.constant(4.0)))
    computed_z = sess.run(z)

    y = tf.linspace(-3.0, 3.0, 100)
    computed_y = sess.run(y)
    computed_z2 = sess.run(z)

    sess.close() # Close the session

    computed_z
    computed_z2
    computed_y


#### Tensorflow Variables
    # https://www.tensorflow.org/api_docs/python/tf/Variable

    import tensorflow as tf
    A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print(sess.run(A[:2, :2]))  # => [[1,2], [4,5]]

      op = A[:2,:2].assign(22. * tf.ones((2, 2)))
      print(sess.run(op))  # => [[22, 22, 3], [22, 22, 6], [7,8,9]]

    w = tf.Variable(4.0, name="w")

    with tf.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      
      z = tf.add(tf.divide(4.0, 2.0), tf.multiply(3.0, w))
      computed_z = sess.run(z)

      w.assign(10)
      computed_z2 = sess.run(z)


#### Why Tensorflow uses a computational graph?

* All other machine learning frameworks are also based on a
  computational graph, such as: Caffe, Pytorch, Theano, etc.
* A Computational graph has a forward step to compute a prediction
  and backward step to compute the gradients. This graph enables
  `Backpropagation` to minimize the error loss function.
* We can split the graph to run with multiple GPUs, CPUs and TPUs.


### Tensorflow tutorial for MNIST

https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-beginner

https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started


#### Reference

* Tensorflow slides:<br>http://web.stanford.edu/class/cs20si/syllabus.html

* Tutorials<br>https://www.tensorflow.org/tutorials/
* Guide<br>https://www.tensorflow.org/guide/
* Youtube playlist: Edureka! Deep Learning with Tensorflow Videos<br>
https://www.youtube.com/playlist?list=PL9ooVrP1hQOEX8BKDplfG86ky8s7Oxbzg


### Sequence Modeling

* Standard Neural Networks are of fixed length input and fixed length output
* How do we handle variable length inputs and/or variable length outputs?
* Use cases: text generation, music generation, machine translation
* Answer: RNN = Recurrent Neural Networks!


### RNN

The Unreasonable Effectiveness of Recurrent Neural Networks<br>
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks


### RNN architectures

![RNN achitectures](img/rnn-architectures.jpg)
##### Source: [Karpathy's blog: The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)


#### Min-char-rnn, character-level RNN

Karpathy<br>
https://gist.github.com/karpathy/d4dee566867f8291f086

Others:<br>
https://gist.github.com/vinhkhuc/7ec5bf797308279dc587
https://gist.github.com/muggin/3097e7ed45a75dd53bd96c0e430a2895


### char-rnn

* https://github.com/karpathy/char-rnn
* char-rnn: same as min-char-rnn, BUT using LSTM and running with GPU
* char-rnn (lstm, 3-levels, 700 nodes, 1 GPU, 1 day)
* 700 x 3 cells

* Nicer version? Justin Johnson<br>
https://github.com/jcjohnson/torch-rnn


### RNN Examples

* Basic Cheese Wings (dessert)<br>
https://gist.github.com/nylki/1efbaa36635956d35bcc

* Tiny Shakespeare<br>
  - https://github.com/crazydonkey200/tensorflow-char-rnn
  - https://github.com/sherjilozair/char-rnn-tensorflow

* Visual Analysis for Recurrent Neural Networks<br>
http://lstm.seas.harvard.edu/


### Image captioning

* NeuralTalk (Slow, Python + Numpy, for educational reasons)
https://github.com/karpathy/neuraltalk

* NeuralTalk 2
https://github.com/karpathy/neuraltalk2

* Demos
  - https://cs.stanford.edu/people/karpathy/neuraltalk2/demo.html
  - https://vimeo.com/146492001


### Google's Show and Tell

* https://ai.googleblog.com/2016/09/show-and-tell-image-captioning-open.html
* https://github.com/tensorflow/models/tree/master/research/im2txt
* Requirements:<br>
https://github.com/tensorflow/models/blob/master/research/im2txt/conda-env/ubuntu-18-04-environment.yaml


### Language Models

* Word2vec: king + woman - man = queen
<br>https://jalammar.github.io/illustrated-word2vec/
* Spacy: https://course.spacy.io/
* Machine Translation: https://github.com/tensorflow/nmt
* Eminem-based RNN<br>
https://soundcloud.com/mrchrisjohnson/recurrent-neural-shady
* Char RNN: https://research.google.com/seedbank/seed/charrnn


### Chopin RNN

* http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/
* https://www.youtube.com/watch?v=j60J1cGINX4
* https://github.com/hexahedria/biaxial-rnn-music-composition


### Music RNN

* Danshiebler<br>
  - http://danshiebler.com/2016-08-10-musical-tensorflow-part-one-the-rbm/
  - http://danshiebler.com/2016-08-17-musical-tensorflow-part-two-the-rnn-rbm/
  - https://github.com/dshieble/Music_RNN_RBM

* https://towardsdatascience.com/deep-learning-with-tensorflow-part-3-music-and-text-generation-8a3fbfdc5e9b
* https://github.com/burliEnterprises/tensorflow-music-generator

* Wrapper on danshiebler
https://github.com/llSourcell/Music_Generator_Demo


### Magenta.js

Magenta demos
https://magenta.tensorflow.org/demos/

Drum RNN
* https://gogul09.github.io/software/deep-drum
* https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn

Improv RNN
* https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn


### Tensorflow References

* PDF slides: http://web.stanford.edu/class/cs20si/syllabus.html

* Youtube: Hvass Labs playlist
https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
https://github.com/Hvass-Labs/TensorFlow-Tutorials


### Chatbot Assistant

* https://rasa.com
* https://rasa.com/docs/get_started_step3/
* https://github.com/RasaHQ/starter-pack-rasa-stack

* Amazon Lex<br>
https://aws.amazon.com/lex/
