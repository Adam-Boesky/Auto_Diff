# 1. Introduction
The problem that this software solves is the need for a simple, easy-to-use automatic differentiation (AD) library for our clients. Computing derivatives has essential application in nearly all of STEM, particularly when in need of optimization, in machine learning applications, among other methods. Alternative methods of finding derivatives come with drawbacks such as round-off error in numerical differentiation or inefficiency in symbolic differentiation. Automatic differentiation, on the other hand, is both highly efficient and accurate. 

# 2. Background 
Automatic differentiation is different from symbolic differentiation in that it uses a numeric approach rather than a symbolic approach, which is precise but also easier to compute for complex functions. This is a different strategy because it attacks the problem by dividing up the given function into its constituent elementary functions (i.e. adding, subtracting, exponent, etc.) and computing the derivatives using the chain rule in the given order of operations. 

So, in sum, here are the steps: 
1. Divide the function into a graph that displays how each input variable changes by elementary operations within the function. This is the forward trace. 
2. Then for each of those elementary operations, take the partial derivative with respect to one input variable. 
3. We then calculate the derivative by substituting the values of each partial derivative into the partials of subsequent nodes in our function’s graph. This is possible because by definition the chain rule uses elementary operations of partial derivatives. After finding the partial with respect to one element of the input vector, we end up with one column of the Jacobian. More rigorously, the derivative and the chain rule are defined as: 

![alt text](https://code.harvard.edu/CS107/team13/blob/fcbcf1cd052afc702e94f834ab1b0d1dbe7e35ee/docs/latex.png)

4. We need to make these passes with respect to each independent input variable, each time creating another column of the Jacobian until we have finished.
5. A key point is that we want to be calculating the primal trace and tangent trace not separately but simultaneously, at each step through the graph. 
6. From the columns of this Jacobian, we can use linear combinations to find directional derivatives of our functions related to seed vectors of our choice. 

Note, when calculating the forward trace, we use the properties of dual numbers in order to find the values of the primal and tangent traces in parallel. We encode the real part of the dual number as the primal trace and the dual part as the tangent trace. Using the addition and multiplication properties of the dual numbers, this can be used for the chain rule operations described above.

# 3. How to Use AutoDifferentiation
## 3.1. Distribution
The package will be distributed using PyPI for the final milestone, so a future user of the project will have to run 
		```
		python3 -m pip install AutoDifferentiation'
		```
in order to install the package on their local environment. This operation will also install necessary dependencies for the project found in its metadata, in order to make sure the user can fully use the package by having all requirements installed.

For this milestone, we are explaining the usage of our package below, by cloning our repository locally. 

## 3.2. Using the AutoDiff Class
For a smooth usage of our package at this point(as the project is not yet distributed using PyPI), the following steps should be completed:
1. Clone our repository within the 'code.harvard.edu' organization by running 
```
git clone https://code.harvard.edu/CS107/team13.git
```
This step ensures that the package exists locally.

2. To install all the necessary dependencies, in terminal run:
```
pip3 install -r requirements.txt
```
3. Now, following the specifications mentioned for this milestone, assume you want to create a python file in which to import our module for a specific goal. To start using the package, include the following line in your file:
```
from src.ad import AutoDiff as AD
```
By this, you're important the necessary class. More importantly, this assumes that your file was created at the root of our cloned repository. If creating the file in any other place, then please adjust the path accordingly in the "from src.ad" part of the line above. 

4. Now that the package is imported successfully, we will present a demo of how you can use our functionality. Say you want to compute the first-order derivative of a function with both domain and codomain being the set of real numbers. Let this function be: `f(x)=x^2`. Thus, you will use the following to instantiate an AD object for this function:
- Method 1:
```
# define function separately
def f(x) = 
	return x ** 2
# now give it as an input to the instance of AD
func1 = AD(f)
```

- Method 2:
```
# create AD object directly using the function
func1 = AD(lambda x: x ** 2)
```
Now that you created the AD object, you can use the 2 methods `get_val` and `get_jacobian` to which you can give an input and obtain the value of the function with that value substituted for x and its derivative computed at that value, respectively. More specifically:
```
print(func1.get_val(2)[0])
>>> 4

print(func1.get_jacobian(1)[0])
>>> 2.0
```
Mention: the reason you have to add the 0th index when using the methods is that we plan to create the functionality for more dimensions as well, thus we currently output the results as lists. However, since for this milestone we only deal with scalar results, please add the `...[0]` index at the end of calling a specific method. 

You can also assign the computed values outputted by the methods to other python names and use them in your modules. If you want to access both the `get_val` and `get_jacobian` at the same time, you can call the `forward_mode`:
```
print(func1.forward_mode(1))
>>> ([1],[2.0])
```
Mention: Please note that if you choose to use the `forward_mode` method, this will output a tuple of 2 lists, for the same reason noted above.  

# 4. Software Organization
## 4.1. Directory Structure 

![alt text](https://code.harvard.edu/CS107/team13/blob/7641a28d0fce85fffa92d59bcdfac776b8945c73/docs/directory_tree_m2.png)

## 4.2. Modules and Module Functionality
Our module structure and functionality did not change much from milestone1, as we were able to well predict our needs for the completion of this package. The following describes the current organization and functionality of our modules:
* 'ad' module - Contains class for an automatic differentiation object
* 'dual' module - Contains class that implements the functionality of a dual number
* 'elemental_functions' module - Implements the functionality of elemental functions.

The 'ad' and 'dual' modules will be described more in depth in the 'class overview' part of the documentation, while we offer the description of the 'elemental_functions' module here:

### 'elemental_functions' module
The 'elemental_functions' module contains all of the fundamental operations as separate functions within the module. This module is internal in the implementation of the package, as the user does not directly interact with this module, but it is rather imported by other modules in the package that require this functionality.
For now, only trig functions and exponential are implemented, but others will follow for the final milestone of the project. The essence of the implementation of this class is similar to operator overloading, but we are now defining separate python functions for each elemental function correspondent, where we use the usual 'np' functionality and the chain rule to simulate these for dual numbers. The actual code implementation is as follows:

```
from dual import DualNumber
import numpy as np

def sin(input):
    if isinstance(input, DualNumber):
        #simply take sine of real part, then do chain rule for dual part
        return DualNumber(np.sin(input.real), input.dual*np.cos(input.real))
    else:
        return DualNumber(np.sin(input))

def cos(input):
    if isinstance(input, DualNumber): 
        #simply take cosine of real part, then do chain rule for dual part 
        return DualNumber(np.cos(input.real), input.dual * -1 * np.sin(input.real))
    else: 
        return DualNumber(np.sin(input))

def tan(input): 
    if isinstance(input, DualNumber):
        #simply take tangent of real part, then do chain rule for dual part
        return DualNumber(np.tan(input.real), input.dual * (1 / (np.cos(input.real ** 2))))
    else: 
        return DualNumber(np.tan(input))

def exp(input):
    if isinstance(input, DualNumber):
        return DualNumber(np.e ** input.real, (np.e ** input.real) * input.dual)
    else: 
        return DualNumber(np.e ** input)
```

* **Where do tests live? How are they run? How are they integrated?** - 
1. The tests live in the test_suite directory, as specified above in the directory structure image (this directory is at the root of the team13 repo)
2. The tests are run by navigating into the ‘test_suite’ directory and then running a bash script ‘run_tests.sh’ with the command ‘./run_tests.sh’. This bash script calls pytest on all of the test files in the ‘test_suite’ directory.
3. They are integrated with a GitHub workflow, written in a file called ‘test.yml’ located in ‘.github/workflows/‘. Within that .yml file, we wrote lines such that whenever there is a push, the bash script ‘run_tests.sh’ is run. If the test files do not pass all the pytest calls that .sh file, then the .yml will detect the raised error and reflect the failure on the CI badge.
* **How can someone install your package?** - For the distribution of our project we will use PyPi, by following PEP517 guidelines. This will be implemented for the final milestone, as for now, the project can be installed and used by cloning our repository locally (the exact usage of our project is explained above in the "How to use" part of our documentation).

# 5. Implementation

## 5.1. Core Data Structures
Most of the core, conventional data structures are lists, Numpy arrays, and tuples (although for now we have only implemented the scalar input case, thus not requiring the usage of Numpy arrays, we do plan to use them for the final milestone). For instance, we will be able to store the seed vector in a tuple and the Jacobian can be stored in a Numpy array. The input function will also need to be stored in a tuple (it shouldn’t necessarily be mutable) because it can be vector-valued and thus should have multiple layers. 

We also implemented data structures of our own, as we designed and created the "DualNumber" class, which implements the necessary functionalities for working with operations on DualNumbers. The operator overloading can be seen in the "dual" module for each of the dunder methods necessary for arithmetic operations. 

## 5.2. Class Overview
First of all, we have implemented a few key classes in order to carry out the basic needs of Automatic Differentiation:
1. We implemented a AutoDiff class which will represents the class that the user interacts with, as they define an AD object, whose instance they use to access the jacobian and forward mode using the respective methods.
2. We implemented a Dual class to create the functionalities for the dual numbers (which encode the primal and tangent traces).

### DualNumber Class
The DualNumber class will store the primal and tangent traces with respect to each input variable. For the use of our library, this entails the user defining the variables initial value and first derivative, from which we can calculate the primal and tangent traces at the stage of each elementary operation. The primal trace will be stored in the “real” part of the dual number, while the tangent trace is stored in the “dual” part, or second entry. 

Within this Dual class, we have implemented operator overloading. Since we are going to be storing primal and tangent traces within dual numbers. Similar to how we discussed using operator overloading to specify how to do elementary operations on complex numbers, we also need to specify how to implement addition and multiplication using dual numbers. Though the specific implementations are shown in the included code below, the essence is that we split up the dual number argument into its constituent parts, perform the operations to the given parts as they should be performed for each respective operation, and then store the output as a dual number once again. 

Here is a snippet of the actual code implementation, showing the operator overloading for addition and subtraction. You can also see the important .real and .dual instance attributes that allow the storage and update of the real and dual parts of a DualNumber class object.
```
import numpy as np

class DualNumber:
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
	...
```

### AutoDiff Class
The AutoDiff class has methods to carry out forward pass and calculate the Jacobian for now (we will implement the reverse method and its corresponding method for the final milestone). 
Within the AutoDiff class, we are binding the implemention of the package, by using the functionalities of the other 2 modules: 'elemental_functions' and 'dual' when we evaluate the jacobian, whole forward pass or just substitute a value within the function.  At each level, this entails breaking down the input function into the elementary functions using the operator overloading in the 'dual' class and the 'elemental_functions' module, finally passing the output of the primal and tangent traces of each variable (in the form of a dual number) using the Dual class. For now, the 'get_val', 'get_jacobian', and 'forward_mode' methods output scalars (whose functionality in other projects is still highly valuable - see optimization tasks or the computation of Newton's method), but this will be expanded for the final milestone, as users will be able to work with higher order functions with our package. 

The code below is the actual implementation of the class, where the usage of the other modules is observed, as well as assertion statements, that allow us to control the types of inputs given by the user for each public method. Moreover, we used the 2 instance attributes .function and .values to store the relevant information for the forward mode AD: its functions and value(s) for a certain input. 
```
import numpy as np
from dual import DualNumber 
from elemental_functions import *

class AutoDiff():
    def __init__(self, function):
        self.function = function
        self.values = []

    def get_val(self, val=None):
        if val == None:
            try:
                return [value.real for value in self.values]
            except:
                raise ValueError('No val has been passed into AutoDiff instance yet.')
        else:
            try:
                tracer = DualNumber(val, 1)
                self.values.append(self.function(tracer))
            except:
                raise ValueError('Input value must be either int or float.')

            return [value.real for value in self.values]
            
    def get_jacobian(self, val=None):
        if val == None:
            return [value.dual for value in self.values]
        else:
            self.get_val(val)
            return [value.dual for value in self.values]

    def forward_mode(self, val=None):def fn(x):
    return x**2


if __name__ == "__main__":
    
    ad = AutoDiff(fn)
    print(ad.get_val(0))
    print(ad.get_jacobian())
    print(ad.forward_mode())
        return self.get_val(val), self.get_jacobian(val)
```

## 5.3. Future Features
When it comes to the future features of our implementation, we plan that for the final milestone we will implement the Reverse Mode of the Automatic Differentiation. As discussed in class, the reverse mode is useful for large inputs of functions that we should compute the Jacobian of. We will allow the user to specify the AD method that they prefer in computing the desired functionality probably, and we will implement it internall using a new "ReverseMode" class within the "AD" module. However, this will require other functionality in turn, so all of our modules and methods will be adjusted accordingly, but minimally. 

Furthermore, as explained throughout this documentation for Milestone2, we plan to enrich the functionality of the package overall by allowing the AutoDiff objects to take as input functions with domain and codomain in higher dimensions of real numbers. This will entail the usage of numpy arrays and subsequently storing the jacobian and value of the function in arrays as well in order to support this improved functionality. 

Ideally, the changes for these future features will not imply significant changes to the directory and module structure of our package. The reason for this is that we have already taken into account this future implementation when we designed the overall structure at the beginning of this project, within milestone1.

1. For the reverse mode implementation, as explained above, we consider creating a separate class within the "AD" module that will deal with computing the Reverse Mode (this will subsequently imply a method for the forward pass, a method for getting the value of the function after substitution and a method for getting the reverse mode final result). In terms of other dependencies for this new implementation, we will make use of linear algebra predefined functions from the numpy library (such as transpose). In general, we expect the user to prefer the reverse mode for cases where the number of inputs is much bigger than the number of outputs, while still requesting the forward mode where the number of outputs is much bigger than the number of inputs.

2. For the multiple dimensions input implementation, we will adjust all the methods in the 'AD' module and respective classes, since we need to allow for this feature addition by using numpy arrays. 

## 5.4. Dependencies
We are using external libraries like NumPy, especially because functions like np.sine already provide some operator overloading functionality that reduces overhead. We will also use other predefined numpy functions and data structures for our implementations (their specific usage is explained above). 
 
# 6. License

After careful consideration of possible choices, we chose the MIT license for our automatic differentiation project. The MIT license is a simple, natural choice for our relatively narrow project scope since it allows any subsequent development (including possibility to “use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software”) by another entity, with the conditions of providing the license and the copyright notices in any future copy of the Software. 

Moreover, the choice of the MIT license allows the software to be provided “as is”, without any future accountability for the copyright owners. Given that the intellectual property rights are owned by Harvard University (according to the computer software property policy found at [this website](https://otd.harvard.edu/faculty-inventors/resources/policies-and-procedures/statement-of-policy-in-regard-to-intellectual-property/#computer-software)), then this type of license also ensures the university is not held liable for any kind of error in the Software that might be propagated further.
Since this automatic differentiation software may prove to be an useful tool in many fields and applications, the MIT license allows its free usage and modification, according to future needs of other developers. 
