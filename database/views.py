import json

from django.shortcuts import render
from django.http import HttpResponse


def data(request):
    return render(request, 'data.html')
