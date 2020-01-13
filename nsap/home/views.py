from django.shortcuts import render
#from home import models as pre

# Create your views here.
from . import connection

def index(request):
    return render (request,'home/main.html')

def predictions(request):
    return render (request,'home/home.html')

def result(request):
    company = request.GET['company']
    print(company)
    rows=latestresult(company)
    return render (request,'home/result.html', {'rows': rows,})

def latestresult(company):
    conn = connection.connectDatabase()
    cur = connection.getCur(conn)
    cur.execute("""select * from result  where company = '%s' order by date desc limit 1""" %company)
    row=cur.fetchall()
    conn.commit()
    cur.close()
    print(row)
    return row

