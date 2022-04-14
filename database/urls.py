from django.urls import path


from . import views




urlpatterns = [
    path("",views.show_database,name='show_database')
]