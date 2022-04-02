

import requests

def get_text_classification(text='',ip='10.10.64.190',model_name='my_tc'):
    url = 'http://{}:8950/predictions/{}/'.format(ip,model_name)
    post_data = {
        'data':text
    }
    headers = {

    }
    res = requests.post(url,data=post_data)
    res = eval(res.content)
    print(res)
    return res


if __name__ == '__main__':
    text = 'Bloomberg has decided to publish a new report on global economic situation.'
    get_text_classification(text)