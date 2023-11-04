from src.dual import DualNumber as DN
import numpy as np 

class Test_Dual_Number:
    '''
    Test class for DualNumber implementation
    Functional with pytest
    '''

    def test_init(self):
        dual1 = DN(1, 2)
        dual2 = DN(1.5, 3.4)

        assert dual1.real == 1
        assert dual1.dual == 2
        assert dual2.real == 1.5
        assert dual2.dual == 3.4

    def test_add(self):
        dual1 = DN(1, 2)
        dual2 = DN(2, 3)

        sum1 = dual1 + 7
        assert sum1.real == 8
        assert sum1.dual == 2
        assert isinstance(sum1, DN)

        sum2 = dual1 + dual2
        assert sum2.real == 3
        assert sum2.dual == 5
        assert isinstance(sum2, DN)


        sum3 = dual1 + 1.5
        assert sum3.real == 2.5
        assert sum3.dual == 2
        assert isinstance(sum3, DN)


    def test_sub(self):
        dual1 = DN(7, 4)
        dual2 = DN(2, 3)

        diff1 = dual1 - 4
        assert diff1.real == 3
        assert diff1.dual == 4
        assert isinstance(diff1, DN)

        diff2 = dual1 - dual2
        assert diff2.real == 5
        assert diff2.dual == 1
        assert isinstance(diff2, DN)
    

        diff3 = dual1 - 3.5
        assert diff3.real == 3.5
        assert diff3.dual == 4
        assert isinstance(diff3, DN)

    def test_mul(self):
        dual1 = DN(1, 2)
        dual2 = DN(2, 3)
    
        prod1 = dual1 * 3
        assert prod1.real == 3
        assert prod1.dual == 6
        assert isinstance(prod1, DN)
    
        prod2 = dual1 * dual2
        assert prod2.real == 2
        assert prod2.dual == 7
        assert isinstance(prod2, DN)
    
        prod3 = dual1 * 1.5
        assert prod3.real == 1.5
        assert prod3.dual == 3
        assert isinstance(prod3, DN)
    
    def test_true_div(self):
        dual1 = DN(1, 2)
        dual2 = DN(2, 3)
        
        quot1 = dual1 / 2
        assert quot1.real == 0.5
        assert quot1.dual == 1
        assert isinstance(quot1, DN)

        quot2 = dual1 / dual2
        assert quot2.real == 0.5
        assert quot2.dual == 0.25
        assert isinstance(quot2, DN)

        quot3 = dual1 / 0.25
        assert quot3.real == 4
        assert quot3.dual == 8
        assert isinstance(quot3, DN)
    
    def test_pow(self):

        dual1 = DN(2, 3)
        dual2 = DN(1, 4)
        n1 = 4
        n2 = 3.4

        pow1 = dual1 ** n1
        assert pow1.real == 16
        assert pow1.dual == 96
        assert isinstance(pow1, DN)

        pow2 = dual1 ** n2
        assert np.isclose(pow2.real, dual1.real ** n2, atol = 1E-6)
        assert np.isclose(pow2.dual, (dual1.real ** n2) * n2 * dual1.dual / dual1.real, atol = 1E-6)
        assert isinstance(pow2, DN)

        pow3 = dual1 ** dual2
        assert pow3.real == dual1.real ** dual2.real
        assert pow3.dual == (dual1.real ** dual2.real) * (dual2.dual * np.log(dual1.real) + dual1.dual * dual2.real / dual1.real)
        assert isinstance(pow3, DN)

    def test_radd(self):
        dual1 = DN(1, 2)

        sum1 = 5 + dual1
        assert sum1.real == 6
        assert sum1.dual == 2
        assert isinstance(sum1, DN)

        sum2 = 3.5 + dual1
        assert sum2.real == 4.5
        assert sum2.dual == 2
        assert isinstance(sum2, DN)

    def test_rsub(self):

        dual1 = DN(1, 2)

        diff1 = 8 - dual1
        assert diff1.real == 7
        assert diff1.dual == -2
        assert isinstance(diff1, DN)
        
        diff2 = -8 - dual1
        assert diff2.real == -9
        assert diff2.dual == -2
        assert isinstance(diff2, DN)

    def test_rmul(self):
        dual1 = DN(2, 3)

        prod1 = 5 * dual1
        assert prod1.real == 10
        assert prod1.dual == 15
        assert isinstance(prod1, DN)

        prod2 = 3.5 * dual1
        assert prod2.real == 7
        assert prod2.dual == 10.5
        assert isinstance(prod2, DN)
    
    def test_rtruediv(self):
        dual1 = DN(2, 3)

        quot1 = 6 / dual1
        assert quot1.real == 3
        assert quot1.dual == -18 / 4
        assert isinstance(quot1, DN)

    def test_eq(self):
        dual1 = DN(1, 2)
        dual2 = DN(1, 2)
        dual3 = DN(1, 5)
        mybool1 = dual1 == dual2
        mybool2 = dual1 == dual3

        assert mybool1
        assert not mybool2

    def test_ne(self):
        dual1 = DN(1, 2)
        dual2 = DN(1, 2)
        dual3 = DN(1, 5)
        mybool1 = dual1 != dual2
        mybool2 = dual1 != dual3

        assert not mybool1
        assert mybool2
    
    def test_neg(self):
        dual1 = DN(1, 2)
        negdual = -dual1

        assert negdual.real == -1
        assert negdual.dual == -2

    def test_pos(self):
        dual1 = DN(-1, 2)
        posdual = +dual1
        assert posdual.real == -1
        assert posdual.dual == 2

    def test_lt(self):
        dual1 = DN(1, 2)
        dual2 = DN(-5)

        assert dual2 < dual1
        assert dual1 < 5
        assert -4 < dual1
    
    def test_le(self):
        dual1 = DN(1, 2)
        dual2 = DN(1)
        dual3 = DN(-3, 4)

        assert dual1 <= dual2
        assert dual2 <= dual1
        assert dual1 <= 1
        assert 1 <= dual1
        assert -4 <= dual1
        assert dual3 <= dual1

    def test_gt(self):
        dual1 = DN(3, 4)
        dual2 = DN(2, 7)

        assert dual1 > dual2
        assert dual1 > 1
        assert 5 > dual2

    def test_ge(self):
        dual1 = DN(4, 5)
        dual2 = DN(8, 9)
        dual3 = DN(4, 9)

        assert dual2 >= dual1
        assert dual2 >= 8
        assert dual3 <= dual1
        assert dual1 <= dual3
        assert 8 >= dual2
        assert dual2 <= 8