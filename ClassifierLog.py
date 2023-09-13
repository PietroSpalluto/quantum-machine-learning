from matplotlib import pyplot as plt


class ClassifierLog:
    def __init__(self):
        self.counts = []
        self.values = []

        self.eval_number = 1

    def update(self, count, value):
        # print('evaluation #{}, loss: {}'.format(self.eval_number, value))

        self.counts.append(self.eval_number)
        self.values.append(value)
        self.eval_number += 1
