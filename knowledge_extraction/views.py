import json

from django.shortcuts import render
from django.http import HttpResponse

from knowledge_extraction.utils import get_text_classification_url


def index(request):
    return render(request,'index.html')

def home(request):
    return render(request,'home.html')

def demo(request):
    return render(request,'demo.html')

