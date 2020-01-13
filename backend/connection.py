# -*- coding: utf-8 -*-
"""

"""
import psycopg2


def connectDatabase():
    conn= psycopg2.connect(host="localhost",dbname="postgres",user="postgres",password="sauhar" )
    return conn

def getCur(conn):
    return conn.cursor()
