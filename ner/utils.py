import requests


def get_text_classification_url(text='',ip='10.10.64.190',model_name='my_tc'):
    url = 'http://{}:8950/predictions/{}/'.format(ip,model_name)
    post_data = {
        'data':text
    }

    res = requests.post(url,data=post_data)


    return eval(res.content)