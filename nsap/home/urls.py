# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:42:17 2019
@author: KHANALSAUHAR
"""

from django.conf.urls import url
from home import views

urlpatterns = [
    url(r'^$',views.index,name='index'),
    url(r'^predictions/', views.predictions,name='predictions'),
    url(r'^result', views.result,name='result'),
 
]
