from django.urls import path
from . import views


app_name = 'knowledge_extraction'

urlpatterns = [
    path('', views.index, name='index'),
    path('demo', views.demo, name='demo'),
]