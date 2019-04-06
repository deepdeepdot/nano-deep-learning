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

1) In class `tf`, implement the other three operations: subtract, divide, and multiply

2) Using the class `tf`, rewrite the expression "4 / 2 + 3 * 4"

3) Using the class `tf`, rewrite the expression "3 * 1 + 6 / 3 - 3 + 5"


### HTML

Consider the following recursive definition of some webpage

    webpage = ["html", [
        "head", [
            "title", "Learning the Buddha"
        ],
        "body", [
            ["h1", "The Four Noble Truths")]
            ["ul", [
                ["li", "What's suffering?"],
                ["li", "The cause of suffering"],
                ["li", "The end of suffering"],
                ["li", "A path to the end of suffering"],
            ])
        ])
    ])

1. Write a recursive function to print this hmtl tree
    Hint: from the looks, if the second item is an array, we need to use recursion.

2. Add indentation when printing the html tree


### HTML DOM

1. Implement the dom class with the following API

        dom(tag, id, tagAttributes, text, children)

        webpage = dom("html", children=[
            dom("head", children=[
                dom("title", text="deep learning for dummies")
            ]),
            dom("body", children=[
                dom("h1", text="The four noble truths"),
                dom("ul", children=[
                    dom("li", text="What's suffering?"),
                    dom("li", text="The cause of suffering"),
                    dom("li", text="The end of suffering"),
                    dom("li", text="A path to the end of suffering"),
                ]),
                dom("p", id="last_mark", text="")
            ])
        ])

2. Implement a recursive indented printing function

3. How about supporting the image tag?

        img = dom("img", {
            source: "https://minecraft.net/favicon-32x32.png",
            title: "Minecraft"
        });

4. Implement more APIs to support the following

        webpage.getElementById("last_mark").appendChild(img)

    Print recursively the resulting HTML tree

