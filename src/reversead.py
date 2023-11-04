#this file will import ad elements from the ad object and implement reverse ad
import numpy as np

class ReverseNode():
    def __init__(self, val, gradient=None):
        self.val = val
        self.child_pointers = []
        self.gradient = gradient #we populate this on the second pass
    
    def __add__(self, arg):
        try: 
            val = self.val + arg.val
            gradient = [1,1]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except:
            val = self.val + arg
            gradient = [1]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a
    
    def __radd__(self, arg):
        return self.__add__(arg)

    def __mul__(self, arg):
        try:
            val = self.val * arg.val
            gradient = [arg.val, self.val]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except:
            val = self.val * arg
            gradient = [arg]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a

    def __rmul__(self, arg):
        return self.__mul__(arg)
    
    def __sub__(self, arg):
        try:
            val = self.val - arg.val
            gradient = [1, -1]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except:
            val = self.val - arg
            gradient = [1]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a
    
    def __rsub__(self, arg):
        try:
            val = -1 * self.val + arg.val
            gradient = [-1, 1]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except:
            val = -1 * self.val + arg
            gradient = [-1]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a

    def __truediv__(self, arg):
        try:
            val = self.val / arg.val
            gradient = [1/arg.val, -(self.val/arg.val**2)]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except: 
            val = self.val / arg
            gradient = [1/arg]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a

    def __rtruediv__(self, arg):
        try:
            val = arg.val / self.val
            gradient = [-(arg.val/self.val**2), 1/self.val] 
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except: 
            val = arg / self.val
            gradient = [-(arg/self.val**2)]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a

    def __pow__(self, arg):
        try: 
            val = self.val ** arg.val
            gradient = [arg.val * self.val ** (arg.val - 1), (self.val ** arg.val) * np.log(self.val)]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self, arg]
            return a
        except: 
            val = self.val ** arg
            gradient = [arg * self.val ** (arg - 1)]
            a = ReverseNode(val, gradient)
            a.child_pointers = [self]
            return a
    
    def __eq__(self, arg):
        try: 
            return (self.val == arg.val and self.gradient == arg.gradient)
        except: #should we be asking for an attribute error in these? 
            return (self.val == arg)

    def __ne__(self, arg):
        return not self.__eq__(arg)

    def __neg__(self):
        val = -1 * self.val
        gradient = [-1, self.val]
        a = ReverseNode(val, gradient)
        a.child_pointers = [self]
        return a

    def __pos__(self):
        val = self.val
        gradient = [1, self.val]
        a = ReverseNode(val, gradient)
        a.child_pointers = [self]
        return a

    def __lt__(self, arg):
        try: 
            return self.val < arg.val
        except:
            return self.val < arg

    def __le__(self, arg):
        try:
            return self.val <= arg.val
        except: 
            return self.val <= arg

    def __gt__(self, arg):
        try: 
            return self.val > arg.val
        except: 
            return self.val > arg

    def __ge__(self, arg):
        try: 
            return self.val >= arg.val
        except:
            return self.val >= arg

    def __repr__(self):
        return "{class_name}(val={val}, child_pointers={child_pointers}, gradient={grads})".format(class_name=type(self), val=self.val, child_pointers=self.child_pointers, grads=self.gradient)


if __name__ == "__main__":
    x = ReverseNode(2)
    y = ReverseNode(3)
    z = x*y
    print(z.val)
    print(z.gradient)
    print(z.child_pointers)