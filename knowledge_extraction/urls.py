from django.urls import path


from . import views




urlpatterns = [
    path('get_text_classification',views.get_text_classification,name='get_text_classification'),


    path("ke",views.KnowledgeExtraction.as_view(),name='knowledge_extraction'),

    path("test2",views.Test.as_view(),name='test2')
]