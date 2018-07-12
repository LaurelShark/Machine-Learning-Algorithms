from __future__ import print_function

"""
Training set contains these features:
1 - amount of given credit
2 - gender (1 - male, 0 - female)
3 - education 
4 - marital status
5 - age
6 - 8 - history of payment (July 2005 - September 2005)
9 - 11 - Amount of bill statement
"""

training_data =[[20000, 0, 'university', 'married', 24, 2, 2, -1, 3913, 3102, 689, 'credible'],
               [360000, 0, 'graduate school', 'married', 49, 1, -2, -2, 253286, 246536, 194663, 'not credible'], 
               [120000, 0, 'university', 'married', 39, -1, -1, -1, 316, 316, 316, 'credible'], 
               [450000, 0, 'graduate school', 'married', 40, -2, -2, -2, 5512, 19420, 1473, 'credible'],
               [60000, 1, 'graduate school', 'single', 27, 1, -2, -1, -109, -425, 259, 'credible'],
               [50000,  0, 'high school', 'single', 30, 0, 0, 0, 22541, 16138, 17163, 'not credible'],
               [50000,  0, 'high school', 'married', 47, -1, -1, -1, 650, 3415, 3416, 'not credible'],
               [70000, 1, 'high school', 'single', 42, 1, 2, 2, 37042, 36171, 38355, 'credible'],
               [310000, 0, 'university', 'married', 49, -2, -2, -2, 13465, 7867, 7600, 'not credible'],
               [500000, 0, 'graduate school', 'married', 45, -2, -2, -2, 1905, 3640, 162, 'not credible'],
               [10000, 1, 'university', 'married', 56, 2, 2, 2, 2097, 4193, 3978, 'credible'],
               [210000, 0, 'graduate school', 'single', 30, 2, -1, -1, 300, 300, 1159, 'not credible'],
               [130000, 0, 'high school', 'single', 29, 1, -2, -2, -190, -9850, -9850, 'not credible'],
               [320000, 1, 'university', 'single', 29, 2, 2, 2, 58267, 59246, 60184, 'credible'],
               [200000, 0, 'university', 'married', 32, -1, -1, -1, 9076, 5787, 684,'not credible']
] 

header = ["credit amount", "gender", "education", "marital status", "age", "pay_0", "pay_1", "pay_2", "bill_amt0", "bill_amt1", "bill_amt2", "label"]

"""
training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]

"""
def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

# true_rows, false_rows = partition(training_data, Question(0, 'Green'))
# info_gain(true_rows, false_rows, current_uncertainty)

def find_best_split(rows):
    best_gain = 0  
    best_question = None  
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1 

    for col in range(n_features): 
        values = set([row[col] for row in rows])  # unique values in the column
        for val in values:  
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self, question,true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)



# The tree predicts the 1st row of our
# training data is an apple with confidence 1.
# my_tree = build_tree(training_data)
# classify(training_data[0], my_tree)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


# On the second example, the confidence is lower
# print_leaf(classify(training_data[1], my_tree))


if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)
       # Evaluate
    testing_data = [
    # negative - negative
    [500000, 0, 'university', 'married', 54, -2, -2, -2, 10929, 4152, 22722, 'not credible'],
    # fake example
    # positive - negative
    [1000000, 0, 'high school', 'single', 84, -2, -2, -2, 13522, 6189, 12758, 'credible'],
    # positive - positive
    [360000, 0, 'graduate school', 'married', 45, -1, -1, 2, 390, 1170, 780, 'credible']
    ]

   

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))


 
    

"""
 


    testing_data = [
        ['Green', 8.7, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

"""