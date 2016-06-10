#!/usr/bin/env python
import numpy as np
import pandas as pd


class BaseEstimator(object):
    def __init__(self, model, data, state_names=None, complete_samples_only=True):
        """
        Base class for parameter estimators in pgmpy.

        Parameters
        ----------
        model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
            model for which parameter estimation is to be done

        data: pandas DataFrame object
            datafame object with column names identical to the variable names of the model.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """

        self.model = model
        self.data = data
        self.complete_samples_only = complete_samples_only

        if not isinstance(state_names, dict):
            self.state_names = {node: self._collect_state_names(node) for node in model.nodes()}
        else:
            self.state_names = dict()
            for node in model.nodes():
                if node in state_names:
                    if not set(self._collect_state_names(node)) <= set(state_names[node]):
                        raise ValueError("Data contains unexpected states for variable '{0}'.".format(str(node)))
                    self.state_names[node] = sorted(state_names[node])
                else:
                    self.state_names[node] = self._collect_state_names(node)

    def _collect_state_names(self, variable):
        "Return a list of states that the variable takes in the data"
        states = sorted(list(self.data.ix[:, variable].dropna().unique()))
        return states

    def state_counts(self, variable, complete_samples_only=None):
        """
        Return counts how often each state of 'variable' occured in the data.
        If the variable has parents, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done

        complete_samples_only: bool
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.NaN` somewhere are ignored. If `False` then
            every row where neither the variable nor its parents are `np.NaN` is used.
            Desired default behavior can be passed to the class constructor.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> data = pd.DataFrame(data={'A': ['a1', 'a1', 'a2'],
                                      'B': ['b1', 'b2', 'b1'],
                                      'C': ['c1', 'c1', 'c2']})
        >>> estimator = BaseEstimator(model, data)
        >>> estimator.state_counts('A')
            A
        a1  2
        a2  1
        >>> estimator.state_counts('C')
        A  a1      a2
        B  b1  b2  b1  b2
        C
        c1  1   1   0   0
        c2  0   0   1   0
        """

        parents = sorted(self.model.get_parents(variable))
        parents_states = [self.state_names[parent] for parent in parents]

        # default for how to deal with missing data can be set in class constructor
        if complete_samples_only is None:
            complete_samples_only = self.complete_samples_only
        # ignores either any row containing NaN, or only those where the variable or its parents is NaN
        data = self.data.dropna() if complete_samples_only else self.data.dropna(subset=[variable] + parents)

        if not parents:
            # count how often each state of 'variable' occured
            state_count_data = data.ix[:, variable].value_counts()
            state_counts = state_count_data.reindex(self.state_names[variable]).fillna(0).to_frame()

        else:
            # count how often each state of 'variable' occured, conditional on parents' states
            state_count_data = data.groupby([variable] + parents).size().unstack(parents)

            # reindex rows & columns to sort them and to add missing ones
            # missing row    = some state of 'variable' did not occur in data
            # missing column = some state configuration of current 'variable's parents
            #                  did not occur in data
            row_index = self.state_names[variable]
            column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)

        return state_counts
