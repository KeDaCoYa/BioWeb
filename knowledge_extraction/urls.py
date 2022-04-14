from django.urls import path


from . import views




urlpatterns = [

    path("knowledge_extraction",views.KnowledgeExtraction.as_view(),name='knowledge_extraction'),
    path("test2",views.Test.as_view(),name='test2')
]