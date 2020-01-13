# -*- coding: utf-8 -*-
"""
Created on TUE Jun 25 08:05:33 2019

@author: KHANALSAUHAR
"""

from django.conf.urls import url
from . import views

urlpatterns = [
url(r'^$', views.index,name='index')
   
        ]