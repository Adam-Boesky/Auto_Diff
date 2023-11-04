from dual import DualNumber
from reversead import ReverseNode
import numpy as np

def sin(input):
    if isinstance(input, DualNumber):
        #simply take sine of real part, then do chain rule for dual part
        return DualNumber(np.sin(input.real), input.dual*np.cos(input.real))
    elif isinstance(input, ReverseNode):
        val = np.sin(input.val) 
        gradient = [np.cos(input.val)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.sin(input))

def cos(input):
    if isinstance(input, DualNumber): 
        #simply take cosine of real part, then do chain rule for dual part 
        return DualNumber(np.cos(input.real), input.dual * -1 * np.sin(input.real))
    elif isinstance(input, ReverseNode):
        val = np.cos(input.val)
        gradient = [-np.sin(input.val)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else: 
        return DualNumber(np.cos(input))

def tan(input): 
    if isinstance(input, DualNumber):
        #simply take tangent of real part, then do chain rule for dual part
        return DualNumber(np.tan(input.real), input.dual * (1 / (np.cos(input.real) ** 2)))
    elif isinstance(input, ReverseNode):
        val = np.tan(input.val)
        gradient = [1 / (np.cos(input.val) ** 2)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a 
    else: 
        return DualNumber(np.tan(input))

def exp(input):
    #more chain rule
    if isinstance(input, DualNumber):
        return DualNumber(np.e ** input.real, (np.e ** input.real) * input.dual)
    elif isinstance(input, DualNumber):
        val = np.e ** input.val
        gradient = [(np.e ** input.val) * np.log(input.val)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else: 
        return DualNumber(np.e ** input)

def arcsin(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.arcsin(input.real), input.dual / np.sqrt(1 - (input.real ** 2)))
    elif isinstance(input, ReverseNode):
        val = np.arcsin(input.val)
        gradient = [1/np.sqrt(1-input.val**2)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.arcsin(input.real))
def arccos(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.arccos(input.real), - input.dual / np.sqrt(1 - (input.real ** 2)))
    elif isinstance(input, ReverseNode):
        val = np.arccos(np.arccos(input.val))
        gradient = [-1/np.sqrt(1-input.val**2)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.arccos(input.real))

def arctan(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.arctan(input.real), input.dual / (1 + (input.real ** 2)))
    elif isinstance(input, ReverseNode):
        val = np.arctan(input.val)
        gradient = 1 / (input.val**2 + 1)
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.arctan(input.real))

def sinh(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.sinh(input.real), np.cosh(input.real) * input.dual)
    elif isinstance(input, ReverseNode):
        val = np.sinh(input.val)
        gradient = np.cosh(input.val)
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.sinh(input.real))

def cosh(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.cosh(input.real), np.sinh(input.real) * input.dual)
    if isinstance(input, ReverseNode):
        val = np.cosh(input.val)
        gradient = [np.sinh(input.val)]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.cosh(input.real))

def tanh(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.tanh(input.real), input.dual / (np.cosh(input.real) ** 2))
    elif isinstance(input, ReverseNode):
        val = np.tanh(input.val)
        gradient = [(1/np.cosh(input.val))**2]
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.tanh(input.real))

def logistic(input):
    if isinstance(input, DualNumber):
        return DualNumber(1 / (1 + (np.e ** (- input.real))), 
                          input.dual * (np.e ** (-input.real)) / ((1 + (np.e ** (-input.real))) ** 2))
    elif isinstance(input, ReverseNode):
        val = 1 / (1 + (np.e ** (-input.val)))
        gradient = np.e**(-input.val) / (1 + np.e**(-input.val)**2)
        a = ReverseNode(val, gradient)
        return a
    else:
        return DualNumber(1 / (1 + (np.e ** (- input.real))))

def sqrt(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.sqrt(input.real), input.dual / (2 * np.sqrt(input.real)))
    elif isinstance(input, ReverseNode):
        val = input.val**0.5
        gradient = 0.5*(input.val)**(-0.5)
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.sqrt(input.real))

def log(input, b=np.e):
    if isinstance(input, DualNumber):
        return DualNumber(np.log(input.real) / np.log(b), input.dual / (input.real * np.log(input.real)))
    elif isinstance(input, ReverseNode):
        val = np.log(input.val)
        gradient = 1/input.val
        a = ReverseNode(val, gradient)
        a.child_pointers = [input]
        return a
    else:
        return DualNumber(np.log(input.real) / np.log(b))


