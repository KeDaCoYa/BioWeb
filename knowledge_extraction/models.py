import datetime

from django.db import models

# Create your models here.

class PmidAnnotate(models.Model):
    # pmid作为主键
    pmid = models.CharField(max_length=30,primary_key=True)
    abstract_text = models.TextField(default='')
    update_time = models.DateTimeField(default=datetime.datetime.strptime('1970-01-01', '%Y-%m-%d'))
    entities = models.TextField(default='')
    relations = models.TextField(default='')

    class Meta:
        verbose_name = '标注结果存储'
        verbose_name_plural = 'PMID标注'