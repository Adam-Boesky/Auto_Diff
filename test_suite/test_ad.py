import numpy as np
from src.elementary_functions import sin, cos, tan, exp
from src.ad import AutoDiff as AD

class Test_Auto_Diff():

    @staticmethod
    def fn1(x):
        '''
        x^2 + 1
        '''
        return x ** 2 + 1


    @staticmethod
    def fn2(x):
        '''
        sin(x)
        '''
        return sin(x)

    @staticmethod
    def fn3(x):
        '''
        e^x
        '''
        return exp(x)

    @staticmethod
    def fn4(x):
        '''
        e^(x^2 + 1) + cos(x)
        '''
        return exp((x ** 2) + 1) + cos(x)

    def test_get_val(self):
        func1 = AD(self.fn1)
        func2 = AD(self.fn2)
        func3 = AD(self.fn3)
        func4 = AD(self.fn4)

        assert func1.get_val(0)[0] == 1
        assert func1.get_val(3)[0] == 10
        assert np.isclose(func2.get_val(np.pi / 2)[0], 1, atol=1E-6)
        assert func3.get_val(4)[0] == np.e ** 4
        assert np.isclose(func4.get_val(1)[0], np.e ** 2 + np.cos(1), atol=1E-6)
        assert np.isclose(func4.get_val(0)[0], np.e + 1, atol=1E-6)
    
    def test_get_jacobian(self):
        func1 = AD(self.fn1)
        func2 = AD(self.fn2)
        func3 = AD(self.fn3)
        func4 = AD(self.fn4)

        assert func1.get_jacobian(0)[0] == 0
        assert func1.get_jacobian(2)[0] == 4
        assert np.isclose(func2.get_jacobian(np.pi)[0], -1, atol=1E-6)
        assert np.isclose(func3.get_jacobian(2)[0], np.e ** 2, atol=1E-6)
        assert np.isclose(func4.get_jacobian(2)[0], 4 * (np.e ** 5) - np.sin(2), atol=1E-6)

    def test_forward_mode(self):
        func1 = AD(self.fn1)
        func2 = AD(self.fn2)
        func3 = AD(self.fn3)
        func4 = AD(self.fn4)

        assert [item[0] for item in func1.forward_mode(2)] == [5, 4]

        result1 = [item[0] for item in func2.forward_mode(np.pi / 2)]
        assert np.isclose(result1[0], 1, atol=1E-6)
        assert np.isclose(result1[1], 0, atol=1E-6)

        result2 = [item[0] for item in func3.forward_mode(2)]
        assert np.isclose(result2[0], np.e ** 2, atol=1E-6)
        assert np.isclose(result2[1], np.e ** 2, atol=1E-6)

        result3 = [item[0] for item in func4.forward_mode(2)]
        assert np.isclose(result3[0], (np.e ** 5) + np.cos(2), atol=1E-6)
        assert np.isclose(result3[1], 4 * (np.e ** 5) - np.sin(2), atol=1E-6)