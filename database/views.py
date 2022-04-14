from django.shortcuts import render
# Create your views here.
from django.http import HttpResponse
from django.views import View



def show_database(request):
    return HttpResponse("114514")


