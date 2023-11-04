import numpy as np

class DualNumber:
    '''
    DualNumber class implementation for Automatic Differentiation
    '''

    def __init__(self, real, dual=1):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return "{class_name}(real={real}, dual={dual})".format(class_name=type(self).__name__, real=self.real, dual=self.dual)

    def __add__(self, dual2):
        try:
            return DualNumber(real=self.real+dual2.real, dual=self.dual+dual2.dual)
        except AttributeError:
            return DualNumber(real=self.real+dual2, dual=self.dual)

    def __sub__(self, dual2):
        try:
            return DualNumber(real=self.real-dual2.real, dual=self.dual-dual2.dual)
        except AttributeError:
            return DualNumber(real=self.real-dual2, dual=self.dual)

    def __mul__(self, dual2):
        try:
            return DualNumber(real=self.real*dual2.real, dual=self.real*dual2.dual+self.dual*dual2.real)
        except AttributeError:
            return DualNumber(real=self.real*dual2, dual=self.dual*dual2)

    def __truediv__(self, dual2):
        try:
            return DualNumber(real=self.real/dual2.real, dual=(self.dual * dual2.real - self.real * dual2.dual) / dual2.real**2)
        except AttributeError:
            return DualNumber(real= self.real / dual2, dual= self.dual / dual2) 

    def __pow__(self, n):
        if isinstance(n, float) or isinstance(n, int):
            n = DualNumber(n, 0)
        
        if self.real == 0:
            return DualNumber(0, 0)

        try:
            real = self.real ** n.real
            dual = (self.real ** n.real) * (np.log(self.real) * n.dual + (self.dual * n.real) / self.real)
            return DualNumber(real=real, dual=dual)
        except:
            try: 
                res = DualNumber(self.real, self.dual)
                for _ in range(n.real):
                    res *= self
                return res
            except AttributeError:
                raise TypeError('Raising to an invalid power.')

    def __radd__(self, dual2):
        return self.__add__(dual2)

    def __rsub__(self, dual2):
        return -1 * self + dual2

    def __rmul__(self, dual2):
        return self.__mul__(dual2)

    def __rtruediv__(self, dual2):
        return (self**-1) * (dual2)

    def __eq__(self, dual2):
        equal = False
        try:
            if self.real == dual2.real and self.dual==dual2.dual:
                equal=True
        except AttributeError:
            if self.dual == 0 and self.real == dual2:
                equal=True
        return equal

    def __ne__(self, dual2):
        return not self.__eq__(dual2)

    def __neg__(self):
        return DualNumber(-self.real, -self.dual)

    def __pos__(self):
        return DualNumber(self.real, self.dual)

    def __lt__(self, dual2):
        try:
            return self.real < dual2.real
        except AttributeError:
            return self.real < dual2

    def __le__(self, dual2):
        try:
            return self.real <= dual2.real
        except AttributeError:
            return self.real <= dual2

    def __gt__(self, dual2):
        try:
            return self.real > dual2.real
        except AttributeError:
            return self.real > dual2

    def __ge__(self, dual2):
        try:
            return self.real >= dual2.real
        except:
            return self.real >= dual2

if __name__ == '__main__':
    d = DualNumber(0,1)
    print(d**120)