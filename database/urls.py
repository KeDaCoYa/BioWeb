from django.urls import path
from . import views

app_name = 'database'

urlpatterns = [
    path('', views.data, name='data'),
]