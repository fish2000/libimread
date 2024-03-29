# -*- coding: utf-8 -*-
# Generated by Django 1.10b1 on 2016-06-30 03:35
from __future__ import unicode_literals

from django.db import migrations, models
import im.django.modelfields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Thingy',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('createdate', models.DateTimeField(auto_now_add=True, verbose_name='Created on')),
                ('modifydate', models.DateTimeField(auto_now=True, verbose_name='Last modified on')),
                ('w', models.IntegerField(editable=False, null=True, verbose_name='width')),
                ('h', models.IntegerField(editable=False, null=True, verbose_name='height')),
                ('image', im.django.modelfields.ImageField(height_field='h', max_length=255, null=True, upload_to='images/', verbose_name='Image', width_field='w')),
            ],
            options={
                'abstract': False,
                'verbose_name': 'Image-Hosting Thingy',
                'verbose_name_plural': 'Image-Hosting Thingees',
            },
        ),
    ]
