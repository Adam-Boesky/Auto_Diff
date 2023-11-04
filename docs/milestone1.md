# 1. Introduction
The problem that this software solves is the need for a simple, easy-to-use automatic differentiation (AD) library for our clients. Computing derivatives has essential application in nearly all of STEM, particularly when in need of optimization, in machine learning applications, among other methods. Alternative methods of finding derivatives come with drawbacks such as round-off error in numerical differentiation or inefficiency in symbolic differentiation. Automatic differentiation, on the other hand, is both highly efficient and accurate. 

# 2. Background 
Automatic differentiation is different from symbolic differentiation in that it uses a numeric approach rather than a symbolic approach, which is precise but also easier to compute for complex functions. This is a different strategy because it attacks the problem by dividing up the given function into its constituent elementary functions (i.e. adding, subtracting, exponent, etc.) and computing the derivatives using the chain rule in the given order of operations. 

So, in sum, here are the steps: 
1. Divide the function into a graph that displays how each input variable changes by elementary operations within the function. This is the forward trace. 
2. Then for each of those elementary operations, take the partial derivative with respect to one input variable. 
3. We then calculate the derivative by substituting the values of each partial derivative into the partials of subsequent nodes in our function’s graph. This is possible because by definition the chain rule uses elementary operations of partial derivatives. After finding the partial with respect to one element of the input vector, we end up with one column of the Jacobian. More rigorously, the derivative and the chain rule are defined as 
![alt text](https://code.harvard.edu/CS107/team13/blob/fcbcf1cd052afc702e94f834ab1b0d1dbe7e35ee/docs/latex.png)

4. We need to make these passes with respect to each independent input variable, each time creating another column of the Jacobian until we have finished.
5. A key point is that we want to be calculating the primal trace and tangent trace not separately but simultaneously, at each step through the graph. 
6. From the columns of this Jacobian, we can use linear combinations to find directional derivatives of our functions related to seed vectors of our choice. 

Note, when calculating the forward trace, we use the properties of dual numbers in order to find the values of the primal and tangent traces in parallel. We encode the real part of the dual number as the primal trace and the dual part as the tangent trace. Using the addition and multiplication properties of the dual numbers, this can be used for the chain rule operations described above.

# 3. How to Use AutoDifferentiation
## 3.1. Distribution
The package will be distributed using PyPI, so a future user of the project will have to run 
		```
		python3 -m pip install AutoDifferentiation'
		```
in order to install the package on their local environment. This operation will also install necessary dependencies for the project found in its metadata, in order to make sure the user can fully use the package by having all requirements installed.

Now that the package is installed on the local machine, we will address how an user might physically use the features of the AD package.

## 3.2. Using the AutoDiff Class
To start using the package, the user will include the following line in their module / terminal:

```
import AutoDifferentiation.AutoDiff as ad
```

Assume further that the user might want to compute the derivative of `f(x)=x^2+2*sin(x)at point x = 3`. The first thing that they would need to do is instantiate their input variables as Dual Numbers. For now, we have decided to do it this way because it is easier to directly work with dual numbers in an inputted expression. During the development process of AutoDifferentiation, we plan to change this specification and have the user just input the function, which we will then parse and create the dual numbers internally. 

```
x=Dual(1.0, 1.0) - which is the initial input variable and the derivative; for a multi-dimensional input, we can list these for each variable. I.e. x1, x2 …
test = ad(x^2+2*sin(x))
```

The test will then be an ad object based on the redefined arithmetic operations for dual numbers. Further, to actually compute the derivative of f with respect to x, as in this case this is the only input variable, the user will run the following line:

```
test.diff([0], [3])
```

The first argument to the .diff method is a list of the indices of the input variables with respect to whom we want to calculate the derivative. The second argument specifies the respective points for each variable at which we want to calculate the derivative. 

Internally, when given the task to compute a derivative, the package will decide whether to implement it using forward or reverse mode, mainly based on the number of input variables given by the user. Moreover, the functionality of our package will be much extended (this example is not comprehensive yet, because we haven’t implemented any code so far, thus we provided a general instance of usage for the AutoDiff package), as for instance, we will allow the user to see the general function form of the derivative, as well as obtain the Jacobian matrix for a given function.
The pseudocode for those operations will be similar to the one above, where each respective method will be callable by an instance of the AD class. 

# 4. Software Organization
## 4.1. Directory Structure 

![alt text](https://code.harvard.edu/CS107/team13/blob/b7c8adf1be57c7df89f0d95e2fe0313abde8bdfd/docs/directory_tree.jpeg)

## 4.2. Modules and Module Functionality
* 'ad' module - Contains class for an automatic differentiation object
* 'dual' module - Contains class that implements the functionality of a dual number
* 'operator' module - Contains class that breaks down what to do with elementary functions
* **Where will your test suite live?** - In the Test directory, as specified above in the directory structure image.
* **How will you distribute your package (e.g. 'PyPI' with PEP517/518 or simply setuptools)?** - For the distribution of our project we will use PyPi, by following PEP517 guidelines.

# 5. Implementation

## 5.1. Core Data Structures
Most of the core data structures will be lists, Numpy arrays, and tuples. For instance, the seed vector can be stored in a tuple and the Jacobean can be stored in a Numpy array. The input function will also need to be stored in a tuple (it shouldn’t necessarily be mutable) because it can be vector-valued and thus should have multiple layers.  However, the dual number would be best stored as a unique class because of its unique functionalities (see addition and multiplication properties) and nonreal portion.

## 5.2. Class Overview
First of all, we are going to need to have a few key classes in order to carry out the basic needs of Automatic Differentiation:
1. We need a AD class which will contain a computational graph for a given function
2. We need a Dual class to store the dual numbers (which encode the primal and tangent traces)
3. We need an Operator class the defines the behavior of each elementary operation that we want any given node of our AD to commit

### Operator Class
The Operator class should have all of the fundamental operations (addition, subtraction, multiplication, division, power, sine, log, etc…) as class methods. When declaring the operator class, it will also have a mode attribute to allow for operations to be performed in the forward or backward modes. 

Within this Operator class, we need to be prepared for operator overloading. Since we are going to be storing primal and tangent traces within dual numbers. Similar to how we discussed using operator overloading to specify how to do elementary operations on complex numbers, we also need to specify how to implement addition and multiplication using dual numbers. We also need to specify how to use sine, sqrt, log, and exp. Though the specific implementations can be spelled out in the code, the essence is that we will split up the dual number argument into its constituent parts, perform the operations to the given parts as they should be performed, and then store the output as a dual number once again. 

### Dual Class
The Dual class will store the primal and tangent traces with respect to each input variable. For the use of our library, this entails the user defining the variables initial value and first derivative, from which we can calculate the primal and tangent traces at the stage of each elementary operation. The primal trace will be stored in the “real” part of the dual number, while the tangent trace is stored in the “dual” part, or second entry. 

### AD Class
The AD class will have methods to carry out forward and backward passes, calculate the Jacobian, and get the derivative of the given function. Within the AD class, we are going to call Operator and Dual when we evaluate the forward and backward passes of the graph.  At each level, this entails breaking down the input function into the elementary functions using the Operator class and passing the output of the primal and tangent traces of each variable (in the form of a dual number) using the Dual class.

## 5.3. Future Functionality
Looking forward, in order to accommodate for different types of function objects, we will provide the user with keyword arguments that allow them to indicate the type of function object (i.e. vector/scalar valued) that they are passing in. Our methods will adjust according to this user input.

### 5.4. Dependencies
We do plan to use external libraries like NumPy, especially because functions like np.sine already provide some operator overloading functionality that reduces overhead. 
 
# 6. License

After careful consideration of possible choices, we chose the MIT license for our automatic differentiation project. The MIT license is a simple, natural choice for our relatively narrow project scope since it allows any subsequent development (including possibility to “use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software”) by another entity, with the conditions of providing the license and the copyright notices in any future copy of the Software. 

Moreover, the choice of the MIT license allows the software to be provided “as is”, without any future accountability for the copyright owners. Given that the intellectual property rights are owned by Harvard University (according to the computer software property policy found at [this website](https://otd.harvard.edu/faculty-inventors/resources/policies-and-procedures/statement-of-policy-in-regard-to-intellectual-property/#computer-software)), then this type of license also ensures the university is not held liable for any kind of error in the Software that might be propagated further.
Since this automatic differentiation software may prove to be an useful tool in many fields and applications, the MIT license allows its free usage and modification, according to future needs of other developers. 

# 7. Feedback
## Milestone 1
Introduction(2/2):
This section was really nicely done!

Background(2/2):
This section was really nicely done!

How to use(3/3):
This section was really nicely done!

Software Organization(2/2):
Good

Implementation(4/4):
Good !

License(2/2):

Total: (15/15)
