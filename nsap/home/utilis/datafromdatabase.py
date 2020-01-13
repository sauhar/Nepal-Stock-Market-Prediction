# -*- coding: utf-8 -*-


import psycopg2
def getfromdatabase(company):
    conn= psycopg2.connect(host="localhost",dbname="postgres",user="postgres",password="sauhar" )
    cur = conn.cursor()
    cur.execute("""select openprice,maxprice,minprice,closingprice,date from stockdata where  symbol = '%s' order by date;""" %company)
    row=cur.fetchall()
    conn.commit()
    cur.close()
    #print(row)
    return row

#getfromdatabase()