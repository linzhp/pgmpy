
from pgmpy.models import BayesianModel

import itertools
from collections import defaultdict
import logging
from operator import mul

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DirectedGraph
from pgmpy.factors.discrete import TabularCPD, JointProbabilityDistribution, DiscreteFactor
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.independencies import Independencies
from pgmpy.extern import six
from pgmpy.extern.six.moves import range, reduce
from pgmpy.models.MarkovModel import MarkovModel

class LinBujaBayesianModel(BayesianModel):
    """
    We plan on implementing a continuous model.
 
    What works the same in the discrete case:
    add_edge
    remove_node
    remove_nodes_from
    _get_ancestors_of
    active_trail_nodes
    local_independencies
    is_active_trail
    get_independencies
    to_markov_model works since we are dealing with a directed graph still.
    is_imap(self, JPD):


    What doesn't work automatically:
    get_cardinality - this should throw an error for continuous valued nodes
    check_model - looks at the parents of a CPD
    fit - not doing any learning, no implementing
    predict
    predict_probabilities - needs to return a function ( low necessity to implement)
    get_factorized_product - not needed, not implementing
    copy - since we don't have the cpds implemented properly
    """

    #not implemented
    def predict(self,data):
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> values = pd.DataFrame(np.random.rand(1000,5)*2,
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> train_data = values[:800]
        >>> predict_data = values[800:]
        >>> model = LinBujaBayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> predict_data = predict_data.copy()
        """

        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        # Send state_names dict from one of the estimated CPDs to the inference class.
        model_inference = VariableElimination(self) #passed in the model
        for index, data_point in data.iterrows():
            states_dict = model_inference.query(variables=missing_variables, evidence=data_point.to_dict()) #MAP QUERY IS BROKEN FOR CONTINUOUS FACTOR AS IT CALLS NP.ARGMAX
            for k, v in states_dict.items():
                pred_values[k].append(v) #FIX THIS: only executed once, this is very hacky since k is always equal to zero and we are indexing the default dict
        return pd.DataFrame(pred_values, index=data.index)
        
    def add_cpds(self, *cpds):
        """
        Add CPD (Conditional Probability Distribution) to the Bayesian Model.

        Parameters
        ----------
        cpds  :  list, set, tuple (array-like)
            List of CPDs which will be associated with the model

        EXAMPLE
        -------
        >>> from pgmpy.models import BayesianModel
        >>> import numpy as np
        >>> from scipy.special import beta
        >>> from pgmpy.factors.continuous import ContinuousFactor
        >>> from pgmpy.factors.distributions import CustomDistribution
        # Two variable drichlet ditribution with alpha = (1,2)
        >>> def dirichlet_pdf(x, y):
        ...     return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)
        >>> dirichlet_dist = CustomDistribution(variables=['x', 'y'],distribution=dirichlet_pdf)
        >>> dirichlet_factor = ContinuousFactor(['x', 'y'], dirichlet_dist)
        >>> from pgmpy.factors.discrete.CPD import TabularCPD
        >>> student = BayesianModel([])
        >>> student.add_cpds(dirichlet_factor)
        """
        for cpd in cpds:
            if not isinstance(cpd, TabularCPD) and not isinstance(cpd,ContinuousFactor):
                raise ValueError('Only TabularCPD or ContinuousFactor can be added.')

            if set(cpd.scope) - set(cpd.scope).intersection(
                    set(self.nodes())):
                raise ValueError('CPD defined on variable not in the model', cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning("Replacing existing CPD for {var}".format(var=cpd.variable))
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def copy(self):
        pass

    def get_cardinality(self, node):
        pass

    def fit(self, data, estimator=None, state_names=[], complete_samples_only=True, **kwargs):
        raise TypeError("We have not implementied learning for the continuous variable case.")

    def predict_probabilities(self,data):
        pass