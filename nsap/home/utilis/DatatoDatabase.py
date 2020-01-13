# -*- coding: utf-8 -*-


import urllib.request
from bs4 import BeautifulSoup
import connection

def get_data():
    url = "http://sharesansar.com/c/today-share-price"
    company_list=['ADBL','CHCL','NABIL','NLIC','NTC','OHL','PLIC','SBI','SCB','SHL']
    #company_dictionary={}
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
    for i in range(int(len(list2)/19)):
        b=list2[i*19:(i+1)*19]
        double_list.append(b)
        #print(double_list)	
    conn = connection.connectDatabase()
    for company in double_list:
        if company[2] in company_list:
            insertData(company, conn)
            
           
            #company_dictionary[company[2]] = company
    #return company_dictionary

def insertData(row, conn):
    cur = connection.getCur(conn)
    cur.execute("insert into stockdata(symbol,openprice,maxprice,minprice,closingprice) values (%s,%s,%s,%s,%s)",(row[2],row[3],row[4],row[5],row[6]))
    conn.commit()
    cur.close()
    #print(cur.fetchone())

get_data()   
