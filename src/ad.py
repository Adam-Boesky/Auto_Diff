import numpy as np
from src.dual import DualNumber 
from src.elementary_functions import *
from src.reversead import *


class AutoDiff():
    def __init__(self, function):
        self.function = function

    def get_val(self, vec=None):

        # If a scalar
        if isinstance(vec, (float, int)):
            vec = [vec]
        if isinstance(vec, list):
            vec = np.ndarray(vec)

        # If the user hasn't pass in a list of values
        if not isinstance(vec, np.ndarray):
            raise ValueError('No val has been passed into AutoDiff instance.')

        # If the user is passing in a new val
        else: 
            try:
                # Make tracer vector with all values as DualNumbers and evaluate the given function
                tracer = np.array([DualNumber(val, 1) for val in vec])
                evaluation = self.function(tracer) 
            except:
                raise ValueError('Entries in input vector must be either int or float.')

            return np.array([value.real for value in evaluation])
            
    def get_jacobian(self, vec=None):

        # If a scalar
        if isinstance(vec, (float, int)):
            vec = [vec]
        if isinstance(vec, list):
            vec = np.ndarray(vec)

        # If the user hasn't pass in a list of values
        if not isinstance(vec, np.ndarray):
            raise ValueError('No val has been passed into AutoDiff instance.')
        else:
            # Declare empty matrix for Jacobian
            jacob = []
            for i, val in enumerate(vec):
                lst = []
                for j in range(0, i):
                    lst.append(DualNumber(vec[j], 0))

                lst.append(DualNumber(val, 1))

                for j in range(i + 1, len(vec)):
                    lst.append(DualNumber(vec[j], 0))
                try:
                    curr = self.function(lst)
                except:
                    raise IndexError('Number of inputs do not match number of variables.')
                
                updated = []
                for _, val in enumerate(curr):
                    if not type(val) == DualNumber:
                        updated.append(0)
                    else:
                        updated.append(val.dual)
                jacob.append(updated)

            return np.array(jacob).T

    def forward_mode(self, val=None):
        return self.get_val(val), self.get_jacobian(val)



class ReverseAutoDiff():

    #initialize ReverseAutoDiff object which stores func, dictionary of partials, and root nodes
    #need for all 3 of these becomes clear in the functions below
    def __init__(self, func):
        self.func = func
        self.partials = {}
        self.bases = None

    #function below calculates the partials from the end of the graph back to the roots
    #initial_node is the final node in the graph (counterintuitive but beginning of reverse trace)
    def _get_partials(self, initial_node, trace):

        #for each child of the final node
        for i in range(len(initial_node.child_pointers)):

            #set the trace for each iteration multiply trace by the partial at child (trace starts at 1)
            current_trace = 0
            current_trace = trace*initial_node.gradient[i]

            #using dictionary object in init add to existing trace if child already in dict
            #if not already in dict create new partials dict entry for this trace
            #this is useful because there may be multiple traces leading to root's partial
            if str(id(initial_node.child_pointers[i])) in self.partials.keys():
                self.partials[str(id(initial_node.child_pointers[i]))] += current_trace
            else:
                self.partials[str(id(initial_node.child_pointers[i]))] = current_trace
            
            #call this recursively for the children of the children
            self._get_partials(initial_node.child_pointers[i], current_trace)
    

    def _get_base_partials_1d(self, func_1d, respect_to):
        #used for multidimensional derivative, calculate derivative for each partial for each variable
        self.partials = {str(id(base)): 0 for base in respect_to}
        if type(func_1d) == ReverseNode:
            self._get_partials(func_1d, 1)
            return [self.partials[str(id(base))] for base in respect_to]
        else:
            return [0 for _ in respect_to]


    def get_jacobian(self, vals):

        #run self.partial for each value in the function (single or multidimensional inputs)
        self.bases = [ReverseNode(val) for val in vals]
        graph = self.func(self.bases)
        self.partials = {}
        if len(graph) == 1: # If the function is multivariate
            self._get_partials(self.func, 1)
            return np.array([self.partials[str(id(base))] for base in self.bases])
        else:
            return np.array([self._get_base_partials_1d(func, respect_to=self.bases) for func in graph])


    def get_vals(self, vals):
        self.bases = [ReverseNode(val) for val in vals]
        output = []

        #follow each node in the "forward run" to the end of the graph and find the final output
        for val in self.func(self.bases):
            if not type(val) == ReverseNode:
                output.append(val)
            else:
                output.append(val.val)
        return np.array(output)


    def reverse_mode(self, vals):
        return self.get_vals(vals), self.get_jacobian(vals)



if __name__ == "__main__":

    def f_forward(lst):
        return [lst[0]*lst[1] + 2*sin(lst[0])**2 + 2, 1, lst[2]*lst[1] + log(lst[0], )]
    
    def f_reverse(lst):
        return [lst[0]*lst[1] + 2*sin(lst[0])**2 + 2, 1, lst[2]*lst[1] + log(lst[0], )]

    rad = ReverseAutoDiff(f_reverse)
    ad = AutoDiff(f_forward)
    vals = [2,2,3]
    print('forward: \n', ad.forward_mode(vals)[1])
    print('reverse: \n', rad.reverse_mode(vals)[1])
