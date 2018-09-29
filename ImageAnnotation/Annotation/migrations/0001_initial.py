# Generated by Django 2.0.5 on 2018-09-28 15:55

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Category',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(max_length=20)),
                ('num', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Painting',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('width', models.IntegerField()),
                ('height', models.IntegerField()),
                ('id_list', models.TextField()),
                ('filename', models.CharField(max_length=50)),
                ('picture_url', models.ImageField(upload_to='img/%Y/%m/%d')),
                ('pating_url', models.ImageField(upload_to='pating/%Y/%m/%d')),
            ],
        ),
        migrations.CreateModel(
            name='Psection',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pid', models.IntegerField()),
                ('label', models.CharField(max_length=20)),
                ('sectionname', models.TextField()),
                ('xmin', models.IntegerField()),
                ('ymin', models.IntegerField()),
                ('xmax', models.IntegerField()),
                ('ymax', models.IntegerField()),
                ('score', models.CharField(max_length=10)),
                ('section_url', models.ImageField(upload_to='imgsection/%Y/%m/%d')),
            ],
        ),
    ]