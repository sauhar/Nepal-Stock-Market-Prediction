# -*- coding: utf-8 -*-
"""
Created on TUE Jun 25 17:49:50 2019

@author: KHANALSAUHAR
"""
import psycopg2


def connectDatabase():
    conn= psycopg2.connect(host="localhost",dbname="postgres",user="postgres",password="sauhar" )
    return conn

def getCur(conn):
    return conn.cursor()
