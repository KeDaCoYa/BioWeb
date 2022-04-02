import json

from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from ner.utils import get_text_classification_url


def index(request):
    return render(request,"index.html")


def get_text_classification(request):
    '''
    这是demo展示，文本分类
    '''
    print(request.POST)
    print('----------------------')
    raw_text =  request.POST['raw_text'][0]

    res = get_text_classification_url(text=raw_text)

    # 这是返回json格式(ajax)
    data = {}
    data['success'] = True
    data['res'] = res
    # 以Ajax的形式返回
    return HttpResponse(json.dumps(data, ensure_ascii=False), content_type="application/json;charset=utf-8")



