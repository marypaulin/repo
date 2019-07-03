import lib.vector as vect
from lib.frequency import count

class Tree:

    def __init__(self, capture, table, dataset):
        if not table.has(capture):
            print("Insufficent information for key {}".format(capture))
        result = table.get(capture)
        (split, prediction) = result.optimizer

        risk = result.optimum.value()
        self.risk = risk
        self.split = split
        self.prediction = prediction
        if split != None:
            (left_capture, right_capture) = dataset.split(capture, split)
            self.left_subtree = Tree(left_capture, table, dataset)
            self.right_subtree = Tree(right_capture, table, dataset)

    def predict(self, sample):
        if self.prediction != None:
            return self.prediction
        elif sample[self.split] == 0:
            return self.left_subtree.predict(sample)
        elif sample[self.split] == 1:
            return self.right_subtree.predict(sample)

    def rule_lists(self, dataset):
        if self.prediction != None:
            rule_lists = (
                (
                    ('_',) * dataset.width,
                    "predict {}, risk contribution {}".format(self.prediction, self.risk)
                ),
            )
            return rule_lists
        else:
            left_rule_lists = (
                (
                    tuple('0' if j == self.split else rule_list[0][j] for j in range(dataset.width)), 
                    rule_list[1]
                ) for rule_list in self.left_subtree.rule_lists(dataset))
            right_rule_lists = (
                (
                    tuple('1' if j == self.split else rule_list[0][j] for j in range(dataset.width)),
                    rule_list[1]
                ) for rule_list in self.right_subtree.rule_lists(dataset))
            return tuple(left_rule_lists) + tuple(right_rule_lists)

    def visualize(self, dataset):
        return '\n'.join("({}) => {}".format(','.join(rule_list[0]), rule_list[1]) for rule_list in self.rule_lists(dataset))

