### Computational Graph

In class, we have seen the class Node that supports recursive numeric expressions and some examples of Tensorflow.

operations = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b
}

class Node:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def eval(self, node=None):
        if (node is None): return self.eval(self) # for root.eval()
        if type(node.data) == int or type(node.data) == float:
            return node.data

        value_left = self.eval(node.left)
        value_right = self.eval(node.right)
        return node.data( value_left, value_right )

    # Expression for: "4 / 2 + 3 * 4"
    root = Node(operations['+'],
        left=Node(operations['/'], left=Node(4), right=Node(2)),
        right=Node(operations['*'], left=Node(3), right=Node(4))
    )
    total = root.eval()

#### Part A

The previous example is for the expression: "4 / 2 + 3 * 4"

Implement the expression: "3*1+6/3-3+5".
There are multiple implementations for such implementation.
See if you can come up with all of them, or at least one.


#### Part B

Tensorflow supports tf.add(), tf.subtract, tf.multiply and so on.
Let's support this in our example.

    class tf:
        def add(right, left):
            return Node(operations['+'], right, left)

    root = tf.add(Node(4), Node(5))
    root.eval()

1) In class `tf`, implement the other three operations: subtract, divide, and multiply

2) Using the class `tf`, rewrite the expression "4 / 2 + 3 * 4"

3) Using the class `tf`, rewrite the expression "3 * 1 + 6 / 3 - 3 + 5"

