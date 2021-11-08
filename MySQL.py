#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pymysql
import pandas as pd
import numpy as np
import mathlab
log = mathlab.Langrange(xi, yi, x)
new1 = mathlab.Newton_the_first(x, y)
new2 = mathlab.Newton_the_second(x, y)
#server connection
mydb = pymysql.connect(
  host="localhost",
  user="",
  passwd=""
)

mycursor = mydb.cursor() #cursor created

#creating database with name classdb
mycursor.execute("CREATE DATABASE matrixdb;")

import MySQLdb

from sqlalchemy import create_engine

my_conn = create_engine("mysql+mysqldb://:@localhost/matrixdb") #fill details
log.to_sql(con=my_conn,name='table1',if_exists='append')
new1.to_sql(con=my_conn,name='table1',if_exists='append')
new2.to_sql(con=my_conn,name='table1',if_exists='append')



import pymysql

#server connection
mydb = pymysql.connect(
    host="localhost",
    database='matrixdb',
    user="",
    passwd=""
)

mycursor = mydb.cursor() #cursor created
mycursor.execute('SELECT * FROM massiv')

table_rows = mycursor.fetchall()

df = pd.DataFrame(table_rows)
df

