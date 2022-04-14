# Generated by Django 3.2.12 on 2022-04-07 05:44

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PmidAnnotate',
            fields=[
                ('pmid', models.CharField(max_length=30, primary_key=True, serialize=False)),
                ('abstract_text', models.TextField(default='')),
                ('update_time', models.DateTimeField(default=datetime.datetime(1970, 1, 1, 0, 0))),
                ('entities', models.TextField(default='')),
                ('relations', models.TextField(default='')),
            ],
            options={
                'verbose_name': '标注结果存储',
                'verbose_name_plural': 'PMID标注',
            },
        ),
    ]