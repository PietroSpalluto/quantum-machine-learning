class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []

    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        """Save intermediate results. Optimizer passes five values, but we ignore the last two."""
        print('evaluation #{}, loss: {}'.format(evaluation, cost))

        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)
