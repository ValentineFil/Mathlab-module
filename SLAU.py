#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
import sys
from tkinter.filedialog import asksaveasfilename
import pymysql
 
    
# !!! ВНИМАНИЕ МАТРИЦЫ СОХРАНЯТЬ ПОД ИМЕНЕМ matrix(в десятичном виде) и bad_matrix (в дробном виде)

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


import pandas as pd
import numpy as np
import sympy as sym
df = pd.read_csv('matrix1.csv', delimiter=' ', header = None)
df2 = pd.read_csv('bad_matrix.csv', delimiter=' ', header = None)
print(df)
print(df2)

# Сохраняем в mysql
import pymysql
import pandas as pd
import numpy as np
import mathlab

#server connection ()
mydb = pymysql.connect(
  host="localhost",
  user="root",
  passwd="Valik3612336123"
)

mycursor = mydb.cursor() #cursor created

#creating database 
mycursor.execute("CREATE DATABASE matrix_newdb;")

import MySQLdb

from sqlalchemy import create_engine

my_conn = create_engine("mysql+mysqldb://root:Valik3612336123@localhost/matrix_newdb") #fill details
df.to_sql(con=my_conn,name='table1',if_exists='append')
df2.to_sql(con=my_conn,name='table2',if_exists='append')

import pymysql

#server connection
mydb = pymysql.connect(
    host="localhost",
    database='matrix_newdb',
    user="root",
    passwd="Valik3612336123"
)

mycursor = mydb.cursor() #cursor created
mycursor.execute('SELECT * FROM table1')

table_rows = mycursor.fetchall()

df = pd.DataFrame(table_rows)
print(f' df {df}')

A = df.loc[:, :len(df)-1]
A = np.array(A)
B = df.loc[:, len(df):len(df)]
B = np.array(B)
A1 = df2.loc[:, :len(df2)-1]
A1 = np.array(A1)
B1 = df2.loc[:, len(df2):len(df2)]
B1 = np.array(B1)
x = np.zeros_like(B)

K = np.column_stack((A1, B1))
C = to_proper_fraction(K)
C = np.array(C).astype(np.float32)
A2 = C[:, :len(C)]
B2 = C[:, len(C):]

mathlab.Jacobi(A2, B2, 0.001)
mathlab.Jordan_Gauss(A2, B2)
mathlab.Jordan_Gauss_proper(A1, B1)

# print matrix
print("Matrix:" )
for i in range(A1.shape[0]):
    row = ["{}*x{}" .format(A1[i, j], j + 1) for j in range(A1.shape[1])]
    print(" + ".join(row), "=", B1[i])
print()

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/jacobi')
def index1():
    return f' СЛАУ методом Якоби: {mathlab.Jacobi(A, B, 0.001)}'

@app.route('/gauss')
def index2():
    return f' СЛАУ методом Жордана-Гаусса: {mathlab.Jordan_Gauss(A, B)}'

#@app.route('/care/')
#def index3():
    #name, age, profession = "Jerry", 24, 'Programmer'
    #template_context = dict(name=name, age=age, profession=profession)
    #return render_template('index.html', **template_context)

if __name__ == "__main__":
    app.run()


# In[ ]:




