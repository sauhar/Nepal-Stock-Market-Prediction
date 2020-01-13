# -*- coding: utf-8 -*-
"""
"""

import urllib.request
from bs4 import BeautifulSoup
import psycopg2
import time

def get_data():
    url = "http://sharesansar.com/today-share-price"
    company_list=['ADBL','CHCL','NABIL','NLIC','NTC','OHL','PLIC','SBI','SCB','SHL']
    company_dictionary={}
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    html = response.read()
    soup=BeautifulSoup(html,"lxml")
    d=soup.find_all('td')
    list2=[]
    for i in d:
        a=i.text
        if(len(a) > 0):
            a = a.replace(",","")
        list2.append(a)
        #print(list2)
    count=0
    double_list=[]
    for i in range(int(len(list2)/20)):
        b=list2[i*20:(i+1)*20]
        double_list.append(b)
        #print(double_list)	
    conn = connectDatabase()
    for company in double_list:
        if company[1].replace('\n','') in company_list:
            insertData(company, conn)
            company_dictionary[company[1].replace('\n','')] = company
    return company_dictionary



def connectDatabase():
    conn= psycopg2.connect(host="localhost",dbname="postgres",user="postgres",password="sauhar" )
    return conn

def getCur(conn):
    return conn.cursor()

def insertData(row, conn):
    date = time.strftime('%Y-%m-%d')
    print(date)
    cur = getCur(conn)
    # print(row)
    cur.execute("insert into stockdata(date, symbol,openprice,maxprice,minprice,closingprice) values (%s,%s,%s,%s,%s,%s)",(date, row[1],row[3],row[4],row[5],row[6]))
    conn.commit()
    cur.close()
    # print(cur.fetchone())

get_data()
