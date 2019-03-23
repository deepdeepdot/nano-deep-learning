Image -> Tree
+ Neural Network code for 
+ Slides for Neural Network

--> Exercise with tensorflow calculation?
Tensorflow projects?

RNN -> Text generation + GPT-2
-> Language Model Translation? + Captioning?


### Tree and recursion

* Recursion
* Trees
* Tensorflow
* Tensorflow.js


### Life of a Calculator

- Low-level: Little Man Computer?
https://en.wikipedia.org/wiki/Little_man_computer
https://hacks.mozilla.org/2017/02/a-crash-course-in-assembly/

- High-level:
  * Parser: "4 / 2 + 3 * 4" -> tree
  * Eval: tree -> number


#### What's recursion?

Who is Remus?
Remus is a the brother of Romulus

And who is Romulus?
Romulus is the brother of Remus

StackOverflow?

Conway's Game of Life
http://xahlee.info/math/recursive_game_of_life.html
https://nullprogram.com/blog/2014/06/10/

Mandelbrots Fractals
http://blog.recursiveprocess.com/2014/04/05/mandelbrot-fractal-v2/


#### Recursive definitions

    Fibonacci
    fib(0) = 1
    fib(1) = 1
    fib(n) = fib(n-1) + fib(n-2) # for n > 1

    Python challenge: implement fibonacci

    Factorial
    fact(0) = 1
    fact(1) = 1
    fact(n) = n * fact(n-1) # for n > 1

* Python challenge: implement factorial and fibonacci


#### Recursive defintions: Expressions

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
        operand = expression[0] 
        left = eval_expression(expression[1])
        right = eval_expression(expression[2])
        return operations[operand](left, right)

    print("Total: ", eval_expression(computational_expression))


#### Tree: Node class

    Class Node:
      data
      right (pointer to Node)
      left (pointer to a Node)
      isLeaf(): right and left are null

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
        print(node["data"])
        visit(node["right"])
      
    visit(root)
    # What if we "print" after visiting right and left?


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


### Parser?

* What's wrong with our calculator?
  - It's hardcoded for the expression: "4 / 2 + 3 * 4"
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
    # $ conda install tensorflow-gpu # or tensorflow
    import tensorflow as tf

    sess = tf.Session() # Create a session

    # z = "4 / 2 + 3 * 4"
    z = tf.add(tf.divide(4.0, 2.0), tf.multiply(3.0, 4.0))
    z = tf.add(tf.divide(tf.Constant(4.0), tf.Constant(2.0)),
                tf.multiply(tf.Constant(3.0), tf.Constant(4.0)))
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


#### Reference

* Tensorflow slides: http://web.stanford.edu/class/cs20si/syllabus.html


Tensorflow.js Playlist
https://www.youtube.com/watch?v=HEQDRWMK6yY&list=PLZbbT5o_s2xr83l8w44N_g3pygvajLrJ-

