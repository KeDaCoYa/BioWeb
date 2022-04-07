# -*- encoding: utf-8 -*-
"""
@File    :   function_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 18:00   
@Description :   None 

"""
import datetime
import os
import logging

def get_logger(logs_dir='./outputs/logs',logfile_name=''):
    logger = logging.getLogger('main')

    logger.setLevel(level=logging.INFO)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    year, month, day, hour, minute, secondas = now.year, now.month, now.day, now.hour, now.minute, now.second
    handler = logging.FileHandler(os.path.join(logs_dir,
                                               '{} {}_{}_{} {}:{}:{}.txt'.format(logfile_name, year, month, day,
                                                                                 hour,
                                                                                 minute, secondas)))

    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
