### HTML recursion

Consider the following recursive definition of some webpage

webpage = ["html", [
    ["head",[
        ["title", "Learning the Buddha Way"]
    ]],
    ["body", [
        ["h1", "The Four Noble Truths"],
        ["ul", [
            ["li", "What's suffering?"],
            ["li", "The cause of suffering"],
            ["li", "The end of suffering"],
            ["li", "A path to the end of suffering"]
        ]]
    ]]
]]

1. Write a recursive function to print this hmtl tree<br>
Hint: from the looks, if the second item is an array, we need to use recursion.
The solution needs to be done using Python


2. Add indentation when printing the html tree

Notes: This is the output when I run my implementation of `print_html()`

        In [7]: print_html(webpage) 
                                    
        html
            head
                title: Learning the Buddha
            body
                h1: The Four Noble Truths
                ul
                    li: What's suffering?
                    li: The cause of suffering
                    li: The end of suffering
                    li: A path to the end of suffering


### HTML DOM Tree

Note: Though we are working with an HTML DOM Tree, this is a Python-based exercise. This is not a browser-based Javascript exercise. We are emulating the HTML DOM API using python.

1. Implement the Dom class with the following API

        Dom(tag, id, tagAttributes, text, children)

        webpage = Dom("html", children=[
            Dom("head", children=[
                Dom("title", text="Deep learning for dummies")
            ]),
            Dom("body", children=[
                Dom("h1", text="The four noble truths"),
                Dom("ul", children=[
                    Dom("li", text="What's suffering?"),
                    Dom("li", text="The cause of suffering"),
                    Dom("li", text="The end of suffering"),
                    Dom("li", text="A path to the end of suffering"),
                ]),
                Dom("p", id="last_mark", text="This is the end, my only friend.")
            ])
        ])

2. Implement a recursive indented printing function

        printDom(webpage)

3. How about supporting the image tag?

        img = Dom("img", {
            source: "https://minecraft.net/favicon-32x32.png",
            title: "Minecraft"
        });

4. Implement more APIs to support the following
Extra:

        webpage.getElementById("last_mark").appendChild(img)

    Print recursively the resulting HTML Dom tree


### Computational Graph

In class, we have seen that the class Node supports recursive expressions. We also reviewed some short examples of Tensorflow.

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
            if node is None: return self.eval(self) # for node.eval()

            # Alternative:
            # isLeaf = node.right is None and node.left is None
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

This is an implementation for the expression: "4 / 2 + 3 * 4"

    root = Node(operations['+'],
        left=Node(operations['/'], left=Node(4), right=Node(2)),
        right=Node(operations['*'], left=Node(3), right=Node(4))
    )

Implement the expression: "3*1+6/3-3+5".<br>
There are multiple implementations for the above example.<br>
See if you can come up with all of them, or at least one.


#### Part B

Tensorflow supports tf.add(), tf.subtract, tf.multiply, tf.constant and so on.
Let's support this in our example.

    class tf:
        def add(right, left):
            return Node(operations['+'], right, left)

    root = tf.add(Node(4), Node(5))
    root.eval()

1) In class `tf`, implement the other three operations: subtract, divide, and multiply, and `tf.constant()`

2) Using the class `tf`, rewrite the expression "4 / 2 + 3 * 4"

3) Using the class `tf`, rewrite the expression "3 * 1 + 6 / 3 - 3 + 5"


### Image Captioning

Optional: install and play with NeuraTalk2 (it depends on Lua)
https://github.com/karpathy/neuraltalk2


### Rasa Chatbot AI

Optional: install and execute a chat bot
http://rasa.com



