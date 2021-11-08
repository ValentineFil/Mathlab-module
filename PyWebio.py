#!/usr/bin/env python
# coding: utf-8
# Выбираем, вводим и выводим функции
# In[ ]:


import pandas as pd
import numpy as np
import networkx as nx
from pywebio.output import *
import matplotlib.pyplot as plt
from pywebio.input import *
import json

def put_pin_value(text):
    with use_scope('text_output', clear=True):
        put_text(text)
        
def main():
    answer = radio("Choose one", options=['Interpolation', 'Approximation'], name = 'action')
    if answer[action] == 'Interpolation':
        answer1 = radio("Choose one", options=['Langrange', 'Newton_first', 'Newton_second'], name = 'action1')
        if answer['action1'] == 'Interpolation':
            data = input_group("Введи набор xi, yi, x",[
            input('Input xi', name='xi'),
            input('Input yi', name='yi'),
            input('Input x', name='x')
            ])

        if answer1['action1'] == 'Newton_first':
            data = input_group("Введи набор xi, yi",[
            input('Input xi', name='xi'),
            input('Input yi', name='yi'),
        ])
            
        
        if answer1['action1'] == 'Newton_second':
            data = input_group("Введи набор xi, yi",[
            input('Input xi', name='xi'),
            input('Input yi', name='yi'),
        ])

    use_scope('text_output')
    hold()
    else:
        if answer[action] == 'Approximation':
            answer1 = radio("Choose one", options=['Linear', 'Quadratic', 'Normal'], name = 'action1')
            if answer1['action1'] == 'Linear':
                data = input_group("Введи набор xi, yi, x",[
                input('Input xi', name='xi'),
                input('Input yi', name='yi'),
                input('Input x', name='x')
                ])

            if answer1['action1'] == 'Quadratic':
                data = input_group("Введи набор xi, yi",[
                input('Input xi', name='xi'),
                input('Input yi', name='yi'),
            ])


            if answer1['action1'] == 'Normal':
                data = input_group("Введи набор xi, yi",[
                input('Input xi', name='xi'),
                input('Input yi', name='yi'),
            ])
                
data = mathlab.Langrange(xi, yi, x)
new1_pol = mathlab.Newton_the_first(xi, yi)[1]
new2_pol = mathlab.Newton_the_second(xi, yi)[1]

c = []
for i in range(len(data.values())):
    K = list(data.values())[i]
    K = list(map(int, K.split()))
    c.append(K)

df = pd.DataFrame(c)  
df = np.array(df)

xi = df[0]
yi = df[1]
x= df[2]




    
with use_scope(''): 
    (put_text(f' xi, yi, x: {df}')) 

with use_scope(''): 
    (put_text(f' Вывод xi, yi, fi: {lag_pol[1]}')) 
    
with use_scope(''): 
    (put_text(f' xi, yi, x: {df}')) 

with use_scope(''): 
    (put_text(f' Вывод xi, yi, fi: {new1_pol[1]}')) 
    
    
with use_scope(''): 
    (put_text(f' xi, yi, x: {df}')) 

with use_scope(''): 
    (put_text(f' Вывод xi, yi, fi: {new2_pol[1]}'))
    
    

