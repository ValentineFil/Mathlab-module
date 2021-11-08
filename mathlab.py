#!/usr/bin/env python
# coding: utf-8

# In[5]:

# Determination
def det(matrix, j=0): 
    import numpy as np
    # проверям матрицу на вырожденность
    try:
        mt_inv = np.linalg.inv(matrix)
    except:
        print(np.linalg.LinAlgError(matrix))
        print('Singular matrix')
        
    def matrix_2(my_matrix, row, col):
        matrix_1 = my_matrix
        matrix_1 = np.delete(matrix_1, col, axis = 1)
        matrix_1 = np.delete(matrix_1, row, axis = 0)
        return matrix_1
    
    def minor(matx):
        return matx[0][0]*matx[1][1] - matx[1][0]*matx[0][1]
    
    matrix_new = matrix
    num_cols = len(matrix)
        
    # делаем проверку, квадратная ли матрица
    #try:
        #if (len(matrix[0])!=len(matrix)):
            #raise UnAcceptedValueError
    #except:
        #print('матрица не квадратная')
   
    
    if len(matrix_new)== 2:
        return minor(matrix_new)

    else:
       
        deter = 0
        
        for i in range(num_cols):
            minor = (-1)**(i + j)*matrix[i][j] * det(matrix_2(matrix, i, j))
            deter += minor
            
        return deter
    
#==============================================================================================================================
#==============================================================================================================================

#ДЛЯ 2-Х МАТРИЦ
#транспонированная матрица    
import numpy as np    
def TS(matrix):
    import numpy as np
    res = np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])
    return res        
        
#==============================================================================================================================
# Умножение на скаляр       
def MS(matrix, k):
    return np.array([[k*matrix[i][j] for j in range (len(matrix))] for i in range (len(matrix[0]))])
           

#=================================================================================================================================
# Умножение матриц друг на друга
def M(matrix1, matrix2):
    
    assert (len(matrix1[0]) ==len(matrix2)), 'нельзя умножить'
           
    
    matrix3 = np.array([[0 for i in range (len(matrix1[0]))] for j in range (len(matrix2))])
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix1[0])):
                matrix3[i,j] += matrix1[i, k] *matrix2[k,j]
    print(matrix3)  
    
    
#================================================================================================================================
#================================================================================================================================
# N МАТРИЦ
# Транспонирование n матриц
def T(list_matrix):
    T_list_matrix = []
    for matrix in list_matrix:
        res = np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])
        T_list_matrix.append(res)
    return T_list_matrix
                    
#================================================================================================================================
# Умножение друг на друга
from copy import*
def dot(list_matrix, n):
    # n - количество матриц, которые надо перемножить
    # создаем пустой массив для умножения всех n 
    empty_matrix_list = np.array([[0 for i in range (len(list_matrix[0][0]))] for j in range (len(list_matrix[1][1]))])
    for i, x in enumerate(empty_matrix_list):
        x[i]=1
    # проверяем, можно ли умножить
    for t in range(len(list_matrix[:n])):
        assert (len(empty_matrix_list[0]) == len(list_matrix[t])), 'нельзя умножить'
                  
        # создаем нулевой массив для умножения 2-х
        matrix3 = np.array([[0 for i in range (len(empty_matrix_list[0]))] for j in range (len(list_matrix[t]))])
        for i in range(len(empty_matrix_list)):
            for j in range(len(list_matrix[t])):
                for k in range(len(empty_matrix_list[0])):
                    matrix3[i,j] += empty_matrix_list[i, k] * list_matrix[t][k, j]
        # обновляем путую матрицу           
        empty_matrix_list = deepcopy(matrix3)
    return  empty_matrix_list
    
#================================================================================================================================
# Умножение на скаляр

def multiply(list_matrix, n, k):
    # k -число, на которое надо умнодить матрицу
    # n - количество матриц
    empty_matrix_list = []
    for matrix in list_matrix[:n]:
        empty_matrix_list.append(np.array([[k*matrix[i][j] for j in range (len(matrix[0]))] for i in range (len(matrix))]))
    return empty_matrix_list
#=================================================================================================================================
# Деление матриц

def divide(list_matrix, n, k):
    # k -число, на которое надо разделить матрицу
    # n - количество матриц
    empty_matrix_list = []
    for matrix in list_matrix[:n]:
        empty_matrix_list.append(np.array([[matrix[i][j]/3 for j in range (len(matrix[0]))] for i in range (len(matrix))]))
    return empty_matrix_list
    
#==================================================================================================================================
# Минимальный элемент конкретной матрицы

import sys
def mininmum(list_matrix, number):
    list_matrix = np.array(list_matrix)
    # number - номер матрицы в списке
    min_ =  sys.maxsize
    print(min_)
    matrix = list_matrix[number]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] < min_:
                min_ =  matrix[i][j]
    return min_
#====================================================================================================================================
# Максимальный элемент конкретной матрицы

import sys
def maximum(list_matrix, number): 
    list_matrix = np.array(list_matrix)
    # number - номер матрицы в списке
    max_ =  -sys.maxsize
    matrix = list_matrix[number]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > max_:
                max_ =  matrix[i][j]
    return max_
#====================================================================================================================================
# Сложение матриц
def add(list_matrix, n):
    # n - количество матриц, которые надо сложить
    empty_matrix_list = np.array([[0 for i in range (len(list_matrix[1][0]))] for j in range (len(list_matrix[0]))])
    for matrix in list_matrix[:n]:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                empty_matrix_list[i][j] +=  matrix[i][j]
    return empty_matrix_list

#====================================================================================================================================

# Вычитание матриц
def subtract(list_matrix, n, number):
    # выбираем матрицу, из которой нужно вычесть все остальные - number
    empty_matrix_list = list_matrix[number]
    
    for t in range(len(list_matrix)):
        if t != number:
            for i in range(len(list_matrix[t])):
                for j in range(len(list_matrix[t][0])):
                    empty_matrix_list[i][j] -=  list_matrix[t][i][j]
                    
    return empty_matrix_list



#=====================================================================================================================================

#=====================================================================================================================================

#=====================================================================================================================================
#МЕТОД ЯКОБИ
import sys
def Jacobi(a,b,e): 
    # проверяем матрицу на вырожденность
    try:
        mt_inv = np.linalg.inv(a)
    except:
        print(np.linalg.LinAlgError(a))
        print('Singular matrix')
    def isCorrectArray(a):
     # делаем проверку, квадратная ли матрица
        try:
            if (len(a[0])!=len(a)):
                raise UnAcceptedValueError
        except:
            print('матрица не квадратная')
    
        for row in range(0, len(a)):
            if( a[row][row] == 0 ):
                print('Нулевые элементы на главной диагонали')
                
    isCorrectArray(a)
    
   # делаем проверку на диагональное преобладание       
    def diagonal_dominant(A):
        sum_not_diag = np.array([0 for i in range (len(A[0]))])
        sum_diag = np.array([0 for i in range (len(A[0]))])
        
        for i in range(len(A)):
            for j in range(len(A[i])):
                if i != j:
                    sum_not_diag[i]+=abs(A[i][j])
                else:
                    sum_diag[i]+=abs(A[i][j])
     
        for k in range(len(sum_diag)):
            f = True
            if sum_diag[k] < sum_not_diag[k]:
                f = False
        if f == False:
            print('Условие диагонального преобладания не выполняется')
        return f
    
    print("Matrix:" )
    for i in range(a.shape[0]):
        row = ["{}*x{}" .format(a[i, j], j + 1) for j in range(a.shape[1])]
        print(" + ".join(row), "=", b[i])
    
    # находим детерминант матрицы
    det1 = det(a)
    h = np.eye(a.shape[0]) # единичная матрица
    
    # Транспонируем матрицу    
    Tm = TS(a) 
    
    # Вычеркивание строки и столбца
    def matrix_2(my_matrix, row, col):
        matrix_1 = my_matrix
        matrix_1 = np.delete(matrix_1, col, axis = 1)
        matrix_1 = np.delete(matrix_1, row, axis = 0)
        return matrix_1
    
    # основной код - находим СЛАУ и обратную матрицу
    if diagonal_dominant(a):
        q = sys.maxsize # наша погрешность
        x = np.zeros_like(b)
        A1 = np.zeros_like(a)
        inter = 0
        while q>e:
            #print("Current solution:" , x)
            x_new = np.zeros_like(x)
            abs_x_new = np.zeros_like(x)
            for i in range(a.shape[0]):
                s1 = np.dot(a[i, :i], x[:i])
                s2 = np.dot(a[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i]-s1-s2)/a[i,i]
                abs_x_new[i] = abs(x_new[i]-x[i])
                
                for j in range(len(Tm)):
                    minor = (-1)**(i + j)* det(matrix_2(Tm, i, j))
                    A1[i][j] = minor
            inter += 1
            if inter > 1000:
                break
            q = max(abs_x_new)   
            #print(f'q= {q}')
            x = x_new  
            
        reverse_a = A1/det1 #обратная матрица
        
        # вычисляем нормы
        # норма 1
        summa1 = np.zeros_like(a[0])
        summa2 = np.zeros_like(a[0])
        summa3 = 0
        reverse_summa1 = np.zeros_like(a[0])
        reverse_summa2 = np.zeros_like(a[0])
        reverse_summa3 = 0
        for k in range(a.shape[0]):
            for k2 in range(len(a[k])):
                summa1[k] += abs(a[k][k2])
                summa3 += a[k][k2]**2
                reverse_summa1 += abs(reverse_a[k][k2])
                reverse_summa3 += reverse_a[k][k2]**2
               
        n1 = max(summa1)  
        reverse_n1 = max(reverse_summa1)
        
        # норма 2
        for k in range(a.shape[0]):
            for k2 in (a[:, k:k+1]):
                summa2[k] += abs(k2)
                
        for k in range(reverse_a.shape[0]):
            for k2 in (reverse_a[:, k:k+1]):
                reverse_summa2[k] += abs(k2)
            

        n2 = max(summa2)
        reverse_n2 = max(reverse_summa2)
        
        # норма 3
        n3 = summa3**0.5
        reverse_n3 = reverse_summa3**0.5
        
        # считаем число обусловленности
        cond_number1 = n1*reverse_n1
        cond_number2 = n2*reverse_n2
        cond_number3 = n3*reverse_n3
        cond_number = max(cond_number1, cond_number2, cond_number3)
        f = True
        if cond_number >= 1000 or cond_number <1:
            print(f'матрица плохо обусловлена {cond_number}')
            f = False
        else:   
            if (cond_number <= 100) or (cond_number >= 1):
                print(f'матрица хорошо обусловлена {cond_number}')
           
        
            
        return f' норма матрицы: {n3}', f' норма обратной матрицы: {reverse_n3}, решение:', x, reverse_a, f
    
#=========================================================================================================================
# Жордан-Гаусс
def Jordan_Gauss(a, b):
    # проверяем матрицу на вырожденность
    #try:
        #mt_inv = np.linalg.inv(a)
    #except:
        #print(np.linalg.LinAlgError(a))
        #print('Singular matrix')
    
    ones = np.eye(a.shape[0])
    # объединяем данную матрицу и единичную матрицу
    big_matrix = np.column_stack((a, ones))
    matrix = np.column_stack((a, b))
    j = len(matrix[0])
    
    # выводим матрицу
    print("Matrix:" )
    for i in range(a.shape[0]):
        row = ["{}*x{}" .format(a[i, j], j + 1) for j in range(a.shape[1])]
        print(" + ".join(row), "=", b[i])
    
    for i in range(matrix.shape[0]):
        # проверяем, что делим не на 0
        if a[i][i] == 0.0:
            sys.exit('Divide by zero')
            
            
        for j in range(matrix.shape[1]-1, -1, -1):
            
            matrix[i][j] = matrix[i][j]/matrix[i][i]
            devision = matrix[i][j]/matrix[i][i]
           
            
        for j in range(matrix.shape[1]-1, -1, -1):
            
            matrix[i][j] = matrix[i][j]/matrix[i][i]
            devision = matrix[i][j]/matrix[i][i]
            
            for k in range(matrix.shape[0]):
                if (k != i):
                    matrix[k][j] = matrix[k][j] - devision*matrix[k][i]
                    
        # ищем обратную матрицу методом Жордана-Гаусса
        
        j = big_matrix.shape[1]
        
        for j in range(big_matrix.shape[1]-1, -1 + i, -1):
            
            big_matrix[i][j] = big_matrix[i][j]/big_matrix[i][i]
            devision = big_matrix[i][j]/big_matrix[i][i]
            
        for j in range(big_matrix.shape[1]-1, -1 + i, -1):
            
            big_matrix[i][j] = big_matrix[i][j]/big_matrix[i][i]
            devision = big_matrix[i][j]/big_matrix[i][i]
            
            for k in range(big_matrix.shape[0]):
                if (k != i):
                    big_matrix[k][j] = big_matrix[k][j] - devision*big_matrix[k][i]    
                    
         
    reverse_a = big_matrix[:, matrix.shape[0]:]
    #print(f'reverse_a {reverse_a}')
    # вычисляем нормы
    # норма 1
    summa1 = np.zeros_like(a[0])
    summa2 = np.zeros_like(a[0])
    summa3 = 0
    reverse_summa1 = np.zeros_like(a[0])
    reverse_summa2 = np.zeros_like(a[0])
    reverse_summa3 = 0
    for k in range(a.shape[0]):
        for k2 in range(len(a[k])):
            summa1[k] += abs(a[k][k2])
            summa3 += a[k][k2]**2
            reverse_summa1 += abs(reverse_a[k][k2])
            reverse_summa3 += reverse_a[k][k2]**2
               
    n1 = max(summa1)  
    reverse_n1 = max(reverse_summa1)
        
    # норма 2
    for k in range(a.shape[0]):
        for k2 in (a[:, k:k+1]):
            summa2[k] += abs(k2)
                
    for k in range(reverse_a.shape[0]):
        for k2 in (reverse_a[:, k:k+1]):
            reverse_summa2[k] += abs(k2)
            

    n2 = max(summa2)
    reverse_n2 = max(reverse_summa2)
        
    # норма 3
    n3 = summa3**0.5
    reverse_n3 = reverse_summa3**0.5
        
    # считаем число обусловленности
    cond_number1 = n1*reverse_n1
    cond_number2 = n2*reverse_n2
    cond_number3 = n3*reverse_n3
    cond_number = max(cond_number1, cond_number2, cond_number3)
    f = True
    if cond_number >= 1000 or cond_number <1:
        print(f'матрица плохо обусловлена {cond_number}')
        f = False
    else:        
        if (cond_number <= 100) or (cond_number >= 1):
            print(f'матрица хорошо обусловлена {cond_number}')              
                    
                
    return matrix[:, matrix.shape[0]:matrix.shape[1]], f' норма матрицы {n3}', f' норма обратной матрицы {reverse_n3}', f 

#================================================================================================================================

# преобразуем строковый массив в массив с обыкновенными дробами
import sympy as sym
def to_proper_fraction(massiv):

    A1 = np.zeros_like(massiv)
    for i in range(len(massiv)):
        for j in range(len(massiv[0])):
            if '/' in massiv[i][j]:
                if massiv[i][j][0] == '-':
                    left_digit = int(massiv[i][j][1:massiv[i][j].index('/')])
                    right_digit = int(massiv[i][j][massiv[i][j].index('/')+1:])
                    fraction = sym.Rational(left_digit, right_digit)
                    A1[i][j] = fraction
                else:
                    left_digit = int(massiv[i][j][:massiv[i][j].index('/')])
                    right_digit = int(massiv[i][j][massiv[i][j].index('/')+1:])
                    fraction = sym.Rational(left_digit, right_digit)
                    A1[i][j] = fraction
            else:
                A1[i][j] = int(massiv[i][j])
    return A1

# Жордан-Гаусс с обыкновенными дробями
def Jordan_Gauss_proper(a, b):
    # проверяем матрицу на вырожденность
    #try:
        #mt_inv = np.linalg.inv(a)
    #except:
        #print(np.linalg.LinAlgError(a))
        #print('Singular matrix')
    ones = np.eye(a.shape[0])
    # объединяем данную матрицу и единичную матрицу
    matrix = np.column_stack((a, b))
    matrix = to_proper_fraction(matrix)
    a = to_proper_fraction(a)
    big_matrix = np.column_stack((a, ones))
    j = len(matrix[0])
    
    print("Matrix:" )
    for i in range(a.shape[0]):
        row = ["{}*x{}" .format(a[i, j], j + 1) for j in range(a.shape[1])]
        print(" + ".join(row), "=", b[i])
    
    for i in range(matrix.shape[0]):
        # проверяем, что делим не на 0
        if a[i][i] == 0.0:
            sys.exit('Divide by zero')
            
            
        for j in range(matrix.shape[1]-1, -1, -1):
            
            matrix[i][j] = matrix[i][j]/matrix[i][i]
            devision = matrix[i][j]/matrix[i][i]
        
           
            
        for j in range(matrix.shape[1]-1, -1, -1):
            
            matrix[i][j] = matrix[i][j]/matrix[i][i]
            devision = matrix[i][j]/matrix[i][i]
            
            for k in range(matrix.shape[0]):
                if (k != i):
                    matrix[k][j] = matrix[k][j] - devision*matrix[k][i]
                    
        
                    
        # ищем обратную матрицу методом Жордана-Гаусса
        
        
        j = big_matrix.shape[1]
        
        for j in range(big_matrix.shape[1]-1, -1 + i, -1):
            
            big_matrix[i][j] = big_matrix[i][j]/big_matrix[i][i]
            devision = big_matrix[i][j]/big_matrix[i][i]
            
        for j in range(big_matrix.shape[1]-1, -1 + i, -1):
            
            big_matrix[i][j] = big_matrix[i][j]/big_matrix[i][i]
            devision = big_matrix[i][j]/big_matrix[i][i]
            
            for k in range(big_matrix.shape[0]):
                if (k != i):
                    big_matrix[k][j] = sym.Rational(big_matrix[k][j] - devision*big_matrix[k][i])
                
         
    reverse_a = big_matrix[:, matrix.shape[0]:]
    #print(f'reverse_a {reverse_a}')
    # вычисляем нормы
    # норма 1
    summa1 = np.zeros_like(a[0])
    summa2 = np.zeros_like(a[0])
    summa3 = 0
    reverse_summa1 = np.zeros_like(a[0])
    reverse_summa2 = np.zeros_like(a[0])
    reverse_summa3 = 0
    for k in range(a.shape[0]):
        for k2 in range(len(a[k])):
            summa1[k] += abs(a[k][k2])
            summa3 += a[k][k2]**2
            reverse_summa1 += abs(reverse_a[k][k2])
            reverse_summa3 += reverse_a[k][k2]**2
               
    n1 = max(summa1)  
    reverse_n1 = max(reverse_summa1)
        
    # норма 2
    for k in range(a.shape[0]):
        for k2 in (a[:, k:k+1]):
            summa2[k] += abs(k2)
                
    for k in range(reverse_a.shape[0]):
        for k2 in (reverse_a[:, k:k+1]):
            reverse_summa2[k] += abs(k2)
            

    n2 = max(summa2)
    reverse_n2 = max(reverse_summa2)
        
    # норма 3
    n3 = summa3**0.5
    reverse_n3 = reverse_summa3**0.5
        
    # считаем число обусловленности
    cond_number1 = n1*reverse_n1
    cond_number2 = n2*reverse_n2
    cond_number3 = n3*reverse_n3
    cond_number = max(cond_number1, cond_number2, cond_number3)
    f = True
    if cond_number >= 1000 or cond_number < 1:
        print(f'матрица плохо обусловлена {cond_number}')
        f = False
    else:        
        if (cond_number <= 100) or (cond_number >= 1):
            print(f'матрица хорошо обусловлена {cond_number}')              
                
                
    return matrix[:, matrix.shape[0]:matrix.shape[1]], f' норма матрицы {n3}', f'обратная норма {reverse_n3}'

# Langrange
from pywebio.output import *
import matplotlib.pyplot as plt
from pywebio.input import *
import json
def basic_pol(xi, i, x):
    nominator = 1
    denominator = 1
    for j in range(len(xi)):
        if j!=i:
            nominator = nominator*(x - xi[j])
            denominator = denominator*(xi[i] - xi[j])
            polynominal = nominator/denominator
    return polynominal

def Langrange(xi, yi, x):
    polynominals = []
    fi = []
    for i in range(len(xi)):
        polynominals.append(basic_pol(xi, i, x))
    res = 0
    for i in range(len(yi)):
        res += yi[i]*polynominals[i]
        fi.append(yi[i]*polynominals[i])
        
    data = {'xi': xi, 'yi': yi, 'fi': fi}
    res_massiv = pd.DataFrame(data)
    return res_massiv,xi, yi

class Number_decomp():
    def __init__(self, d, fact = False):
        
        # ======
        def primfac(number):
            primfac = {}
            for i in range(2, int(number**(0.5)) + 1):
                while (number % i == 0): 
                    if primfac.get(i, False):
                        primfac[i] += 1
                    else:
                        primfac[i] = 1
                    number //= i
            if (number != 1): 
                primfac[number] = 1
            return primfac
        # ======
        def mul(f, s):
            for elem in s:
                if f.get(elem, False):
                    f[elem] += s[elem]
                else:
                    f[elem] = s[elem]
            
        # ======
        if not(fact):
            n, d = d.as_integer_ratio()
            if d == 1:
                self.decomp = primfac(n)
            else:
                n = primfac(n)
                d = primfac(d)
                d = {key:(-1)*val for key, val in d.items()}
                mul(n, d)
                self.decomp = n

        else:
            assert d % 1 == 0, 'Введите корректное число'
            main = {}
            for i in range(2,d+1):
                mul(main, primfac(i))
            self.decomp = main
    
    def __mul__(self, other):
        under = self.decomp.copy()
        if not(type(other) == Number_decomp):
            other = Number_decomp(other)
        for key, val in other.decomp.items():
            if under.get(key, False):
                under[key] += val
            else:
                under[key] = val
        under = {key:val for key, val in under.items() if val != 0}
        c = Number_decomp(0)
        c.decomp = under
        return c
    
        
    def __rmul__(self, other):
        under = self.decomp.copy()
        if not(type(other) == Number_decomp):
            other = Number_decomp(other)
        for key, val in other.decomp.items():
            if under.get(key, False):
                under[key] += val
            else:
                under[key] = val
        under = {key:val for key, val in under.items() if val != 0}
        c = Number_decomp(0)
        c.decomp = under
        return c
    
    def __truediv__(self, other):
        under = self.decomp.copy()
        if not(type(other) == Number_decomp):
            other = Number_decomp(other)
        for key, val in other.decomp.items():
            if under.get(key, False):
                under[key] -= val
            else:
                under[key] = -val
        under = {key:val for key, val in under.items() if val != 0}
        c = Number_decomp(0)
        c.decomp = under
        return c
    
    @staticmethod
    def C(n,k):
        return Number_decomp(n,fact=True)/(Number_decomp(n-k,fact=True)*Number_decomp(k,fact=True))
    @staticmethod
    def A(n,k):
        return Number_decomp(n,fact=True)/Number_decomp(n-k,fact=True)

    def revers(self):
        numerator, denominator = 1, 1
        for key, val in self.decomp.items():
            if val > 0:
                numerator *= key**val
            else:
                denominator *= key**(-val)
        return numerator, denominator



    def __str__(self):
        return str(self.decomp)
    
from typing import Tuple, List
import bisect

import pandas as pd

# Newton

import copy

 
def coefficient(xi, yi):
    list_coefficients = [] 
    if len(list_coefficients) == 0:
        list_coefficients.append((yi[1] - yi[0])/(xi[1] - xi[0]))
    i = len(list_coefficients) + 1
    while i<(len(xi)):
        ai = ((yi[i] - yi[i-1]) - (yi[i - 1] - yi[i-2]))/Number_decomp(i, True).revers()[0]
        list_coefficients.append(copy.deepcopy(ai))
        i  += 1
    return list_coefficients


def return_P(tk, k):
    multipl = 1
    item = 0
    ki = k
    while item < k:
        multipl *= (tk - ki + 1)
        item += 1
        ki -= 1
    return multipl

def polinom(lst_x, lst_y):
    h = lst_x[1] - lst_x[0]
    r = 1
    lst_raznosty = [lst_y]
    for i in range(len(lst_x)-1):
        l = [-lst_raznosty[r-1][j] + lst_raznosty[r-1][j+1] for j in range(len(lst_x)-r)]
        lst_raznosty.append(l)
        r += 1
    stroka_newton = f'{lst_raznosty[0][0]} +'
    for i in range(1,len(lst_raznosty)):
        a = lst_raznosty[i][0]/(Number_decomp(i, True).revers()[0]*(h**i))
        p = ''
        for j in range(i):
            p += f'*(x - x{j})'
        if a != 0:
            stroka_newton += f'{a}{p} + '
    stroka_newton = stroka_newton[:-2]
    l = stroka_newton.split('+')
    stroka_newton2 = ''
    for el in l[::-1]:
        stroka_newton2 += f'{el} + '
    stroka_newton2 = stroka_newton2[:-2]
    stroka_newton3 = stroka_newton2
    dict_x = {}
    for i in range(len(lst_x)):
        dict_x[f'x{i}'] = lst_x[i]
    
    for key, v in dict_x.items():
        stroka_newton3 = stroka_newton3.replace(f'{key}', f'{v}')
    #print(f' Ньютон: y = {stroka_newton3}')
    return stroka_newton3
  

def Newton_the_first(xi, yi):
    P = 0
    list_P = []
    for i in range(len(xi)):
        if i == 0:
            P += yi[0]
            list_P.append(yi[0])
        else:
            t = (xi[i]-xi[0])/(xi[1]-xi[0])
            P +=  copy.deepcopy(return_P(t, i))*coefficient(xi, yi)[i-1]
            list_P.append(copy.deepcopy(return_P(t, i))*coefficient(xi, yi)[i-1])
    return P, pd.DataFrame(list_P), polinom(xi, yi), print_newton(xi, yi, polinom(xi, yi))
            
def print_newton(lst_x,lst_y, stroka):
    lst_x_new = np.linspace(np.min(lst_x),np.max(lst_x),100)
    lst_y_new = [eval(stroka) for x in lst_x_new]
    plt.plot(lst_x,lst_y,'o',lst_x_new,lst_y_new)
    plt.legend(['данные', 'интер.'], loc = 'best')
    plt.grid(True)
    plt.title(f"Ньютон вперед:  ")
    plt.show()
    print(f"y = {stroka}")
    
    
import copy

 
def coefficient(xi, yi):
    list_coefficients = [] 
    if len(list_coefficients) == 0:
        list_coefficients.append((yi[-1] - yi[-2])/(xi[-1] - xi[-2]))
    i = len(xi) - 1
    n = 1
    while i>n:
        ai = ((yi[i] - yi[i-1]) - (yi[i - 1] - yi[i-2]))/Number_decomp((len(list_coefficients)+1), True).revers()[0]
        list_coefficients.append(copy.deepcopy(ai))
        i  -= 1
    return list_coefficients


def return_P(tk, k):
    multipl = 1
    item = 0
    ki = k
    while item < k:
        multipl *= (tk + ki - 1)
        item += 1
        ki-=1
    
    return multipl

def Newton_the_second(xi, yi):
    P = 0
    list_P = []
    num = len(xi) - 2
    for i in range(num, 0, -1):
        if i == num:
            P += yi[i]
            list_P.append(yi[i])
        else:
            t = (xi[i]-xi[-1])/(xi[-1]-xi[-2])
            P +=  copy.deepcopy(return_P(t, (num - i))*coefficient(xi, yi)[num - i])
            list_P.append(copy.deepcopy(return_P(t, (num-i)))*coefficient(xi, yi)[num-i])
    return P, pd.DataFrame(list_P), polinom(xi, yi), print_newton(xi, yi, polinom(xi, yi))

# In[4]:





# In[ ]:




