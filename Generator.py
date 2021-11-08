#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# ГЕНЕРИРУЕМ И СТРОИМ ГРАФИКИ 
# 1) ВВОДИМОЙ ФУНКЦИИ
# 2) СГЕНЕРИРОВАННЫХ N ФУНКЦИЙ
# 3) КУСОЧНОЙ ФУНКЦИИ

import numpy as np
from numpy import random
random_integer_array = np.random.randint(1, 2000,(1000,2))
print(random_integer_array)
random_integer_array_x = [random_integer_array[i][0] for i in range(len(random_integer_array))]
random_integer_array_y = [random_integer_array[i][1] for i in range(len(random_integer_array))]

def function1(func, interval, step = 0, like = 0):
    
    begin_interval = interval[0]
    end_interval = interval[1]

    # генерируем x

    random_integer_array = np.random.randint(begin_interval, end_interval, generate_num)
    new_random_array = sorted(random_integer_array)
    
    x_means = []
    def take_my_formula(formula, x): 
        try:
            eval(formula)
        except ValueError:
            print("Введи адекватное значение")
            return None
    
        return eval(formula)
    
    # подсчитываем значения и добавляем в список
    for i in range(len(new_random_array)):
        x_means.append(take_my_formula(func, i))
    
    
    return x_means
 import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
import math
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
# для считывания функции из строки
def get_func_from_string(str_func, similarity = False):
    cstepanges = {'^': '**', ',':'.', 'х': 'x'}
    for old_elem, new_elem in cstepanges.items():
        str_func = str_func.replace(old_elem, new_elem)
        
    # если параметр подобия передан - добавляем к X рандомные вещественные коэфициенты от 0.01 до 8
    if similarity:
        count_x = str_func.count('x')
        if count_x:
            last_x_index = str_func.index('x')
        for i in range(count_x):
            new_str = str_func[:last_x_index] + f'{(8 - 0.01) * np.random.random() + 0.01} * x'
            str_func =  new_str + str_func[last_x_index+1:]
            if i != count_x - 1:
                last_x_index = str_func.index('x', len(new_str))
  
    try:
        return eval('lambda x:' + str_func, {'arctan':np.arctan,
                                            'arcsin':np.arcsin,
                                             'arccos':np.arccos,
                                             'tan':np.tan, 
                                             'cos':np.cos, 
                                             'sin':np.sin, 
                                             'log':np.log, 
                                             'e':np.exp,
                                            'sqrt': np.sqrt,
                                            'pi':np.pi})
    except ZeroDivisionError:
        pass
    
# получение координат на заданном интервале с заданными параметрами 
def get_x_y_range(a, b, str_func, fix_step=False, random=False, last_y_value=False, similarity = False):
    len_x = 1000
    if fix_step:
        x_range = np.linspace(a, b, num=len_x, endpoint=True)
    else:
        x_range = np.concatenate(((b - a) * np.random.random(size=len_x-1) + a, [b]))

    func = get_func_from_string(str_func, similarity)
    if not func:
        print('Программа не может работать дальше')
        return None

    y_range = func(x_range)
    # если функция - константа
    if isinstance(y_range, int):
        y_range = np.array([y_range] * x_range.shape[0], dtype=np.float64)
    
    # разброс 10% случайных точек на +-100% от собственных значений
    if random:
        for i in np.random.randint(0, len_x - 1, size = len_x // 10):
            y_range[i] += (2 * np.random.random() - 1) * y_range[i]
    
    if last_y_value:
        y_range = y_range + last_y_value - func(a)
    
    return x_range, y_range, y_range[-1] # y_range[-1] -- last_y_value для следующей функции
#============================================================================================================================
# выспомогательный класс для подсчета факториала функции Пуассона
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


#============================================================================================================================
# ГЕНЕРАЦИЯ N ФУНКЦИЙ. Результат - список списков сгенерированных значений x, y, last_y
# генерируем случайные номера шаблонов
def generate_random(n, a, b, fix_step=False, random=True, last_y_value=False, similarity = False):
    warnings.filterwarnings("ignore", category=FutureWarning)
    templates = ['sin(x)', 'cos(x)', 'tan(x)', 'arctan(x)', 'arcsin(x)', 'arccos(x)', 'x**2','x**3', 'x', 'log(x)', 
                 '1/(x)', "sin(x)/x", "sqrt(x)", '(1/σ*sqrt(2*pi))*e(-((x-μ)**2/(2*σ)**2))', '((λ**k)/k!)*e(-λ)']
    templates_list = []
    k = np.random.randint(0, 14, n)
    functions_points_list = []
    for j in k:
        if templates[j] != '(1/σ*sqrt(2*pi))*e(-((x-μ)**2/(2*σ)**2))' and templates[j] != '((λ**k)/k!)*e(-λ)':
            x, y, last_y = get_x_y_range(min(a,b), max(a,b), templates[j], fix_step, random, last_y_value, similarity)
            functions_points_list.append(np.array([x, y, last_y], dtype=object))
            templates_list.append(templates[j])
           
        else:
            if templates[j] == '(1/σ*sqrt(2*pi))*e(-((x-μ)**2/(2*σ)**2))':
                function = templates[j]
                function = function.replace('σ', f'{np.random.random()}')
                function = function.replace('μ', f'({np.random.randint(-10, 10)})')
                x, y, last_y = get_x_y_range(min(a,b), max(a,b), function, fix_step, random, last_y_value, similarity)
                functions_points_list.append(np.array([x, y, last_y], dtype=object))
                templates_list.append(function)
                
                
            if templates[j] == '((x**k)/k!)*e(-x)':
                function = templates[j]
                function = function.replace('k', f'{np.random.randint(0, 20) }')
                #находим число перед факториалом, и заменям факториал на найденное значение
                end = function.find('!')
                begin = function.find('/')
                l, m = Number_decomp(int(function[begin+1:end]), True).revers()
                function = function.replace(function[begin+1:end+1], f'{l}')
                x, y, last_y = get_x_y_range(min(a,b), max(a,b), function, fix_step, random, last_y_value, similarity)
                functions_points_list.append(np.array([x, y, last_y], dtype=object)) 
                templates_list.append(function)
  
                
    return functions_points_list, templates_list
#print(generate_random(4, -4, 8, random = True, similarity=True))
#===============================================================================================================================
#================================================================================================================================
# ВАРИАНТ 1
# Кусочная функция
def partly_function(func, list_intervals, fix_step=False, random=False, last_y_value=False, similarity = False):
    x_means = []
    y_means = []
    warnings.filterwarnings("ignore", category=FutureWarning)                            
    y_last1 = get_x_y_range(list_intervals[0][0], list_intervals[0][1], func[1], fix_step, random, last_y_value, similarity)[2]
    y_last2 = get_x_y_range(list_intervals[0][0], list_intervals[0][1], func[1], fix_step, random, last_y_value, similarity)[2]               
    if y_last1!= y_last2:
        C = y_last1 - y_last2 
        func[1] = func[1] + f'({C})'
        x1, y1, last_y1= get_x_y_range(list_intervals[0][0], list_intervals[0][1],func[0], fix_step, random, last_y_value, similarity)
        x2, y2, last_y2 = get_x_y_range(list_intervals[1][0], list_intervals[1][1], func[1], fix_step, random, last_y_value, similarity)
        x_means.append(np.array([x1,x2]))
        y_means.append(np.array([y1,y2]))
    else:
        x1, y1, last_y1= get_x_y_range(list_intervals[0][0], list_intervals[0][1],func[0], fix_step, random, last_y_value, similarity)
        x2, y2, last_y2 = get_x_y_range(list_intervals[1][0], list_intervals[1][1], func[1], fix_step, random, last_y_value, similarity)
        x_means.append(np.array([x1,x2]))
        y_means.append(np.array([y1,y2]))
    
    new_x = np.array([np.array(xi) for xi in x_means])
    new_y = np.array([np.array(yi) for yi in y_means])
    for i in new_x:
        new_x = i
    for g in new_y:
        new_y = g    
    return new_x, new_y
#print(partly_function(['2.5*x-1', '11-3.5*x'], [[1,2], [2,3]]))
#================================================================================================================================        
#================================================================================================================================    
#================================================================================================================================
glob_x_range = np.array([])
glob_y_range = np.array([])

# для добавления координат интервала в общий список и послед.записи в цсв
def concatenate(x, y):
    global glob_x_range, glob_y_range
    glob_x_range = np.concatenate((glob_x_range, x))
    glob_y_range = np.concatenate((glob_y_range, y))


# запись в цсв
def write_to_csv(x_range, y_range, sep=',', header=False, index=False, path='C:\\Users\\valen\\Python_lections\\generate1_csv.csv', like = False):
    if like:
        pd.DataFrame({'X': x_range, 'Y': y_range}).to_csv(path_or_buf=path, sep=sep, header=header, index=index)
        
#==================================================================================================================================
#================================================================================================================================
# ГРАФИКИ
# Проверка работы функции
x, y, last_y = get_x_y_range(0, 5,'(x-2)^2', fix_step=True, random=True)
concatenate(x, y)
#plt.plot(x, y,'o', markersize=2)
x, y, last_y = get_x_y_range(5, 10, '2', last_y_value=last_y, random=False)
concatenate(x, y)
#plt.plot(x, y,'o', markersize=2)
x, y, last_y = get_x_y_range(-10, 8, '(1/0.3038670066179413*sqrt(2*pi))*e(-((x-(-10))**2/(2*0.3038670066179413)**2))', last_y_value=last_y, fix_step=True, random=True, similarity = True) 
plt.plot(x, y,'o', markersize=2, label=f'function')
print(plt.show())
#==============================================================================================================================
# N Графиков для сгенерированных функций
import sys
random_functions = (generate_random(3, -4, 8, random = True, similarity=True))[1]
random_functions_means = generate_random(3, -4, 8, random = True, similarity=True)[0]
for func in random_functions:
    warnings.filterwarnings("ignore", category=FutureWarning)
    x, y, last_y = get_x_y_range(10, -4, func, random=True, similarity=True)
    for i in range(len(x)):
        if np.isnan(x[i])== True:
            x[i] = np.random.random() 
        if np.isinf(x[i]) == True:
            x[i] = 0                                                                                          
    concatenate(x, y)
    plt.plot(x, y,'o', markersize=2, label=f' range {func}')
    plt.legend()
    print(plt.show())
#=============================================================================================================================
# Для кусочной функции
x, y = partly_function(['2.5*x-1', '11-3.5*x'], [[1,2], [2,3]]) 
#concatenate(x, y)
plt.plot(x, y,'o', markersize=2, label=f'Кусочная функция')
plt.title(f'Кусочная функция')
print(plt.show())

#==============================================================================================================================

# запись в цсв
write_to_csv(glob_x_range, glob_y_range, header=True)
#log2(x*(19))+(-26)sin(x*(14))+(77)
import tkinter as tk
import sys
from tkinter.filedialog import asksaveasfilename
import pymysql
 
def save_file_as():
    filepath = asksaveasfilename(
        defaultextension = "txt",
        filetypes = [("Текстовые файлы", "*.txt"), ("HTML файлы", "*.html"), ("CSV файлы", "*.csv")]
        )
    if not filepath:
        return
    with open(filepath, "w", encoding = "UTF-8") as output_file:
        text = txt_edit.get("1.0", tk.END)
        output_file.write(text)
    window.title(f"Work with - {filepath}")
    
def exit():
    window.after(1,window.destroy)
    sys.exit

 
window = tk.Tk()
window.title("Save text")
 
window.rowconfigure(0, minsize = 50, weight = 1)
window.columnconfigure(1, minsize = 50, weight = 1)
 
txt_edit = tk.Text(window)
fr_buttons = tk.Frame(window)
fr_buttons2 = tk.Frame(window)


btn_save = tk.Button(fr_buttons, text = "Save file_as", command = save_file_as)
btn_exit = tk.Button(fr_buttons2, text = "Exit", command = exit)

 
btn_save.grid(row = 0, column = 0, padx = 5, pady = 5)
btn_exit.grid(row = 0, column = 0, padx = 5, pady = 5)


fr_buttons.grid(row = 0, column = 0, sticky = "ne")
fr_buttons2.grid(row = 1, column = 0, sticky = "ne")


txt_edit.grid(row = 0, column = 1)
 
window.mainloop()
