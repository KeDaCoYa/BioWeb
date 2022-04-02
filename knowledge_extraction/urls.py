from django.urls import path


from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('get_text_classification',views.get_text_classification,name='get_text_classification'),

]