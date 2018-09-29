from django.db import models

class Painting (models.Model):
    width=models.IntegerField()
    height=models.IntegerField()
    id_list=models.TextField()
    filename=models.CharField(max_length=50)
    picture_url=models.ImageField(upload_to='img/%Y/%m/%d')
    pating_url = models.ImageField(upload_to='pating/%Y/%m/%d')

    def __str__(self):
        return self.filename



class Category(models.Model):
    label=models.CharField(max_length=20)
    num=models.IntegerField()

    def __str__(self):
        return  self.label


class Psection(models.Model):
    pid=models.IntegerField()
    label=models.CharField(max_length=20)
    sectionname=models.TextField()
    xmin=models.IntegerField()
    ymin=models.IntegerField()
    xmax=models.IntegerField()
    ymax=models.IntegerField()
    score=models.CharField(max_length=10)
    section_url=models.ImageField(upload_to='imgsection/%Y/%m/%d')

    def __str__(self):
        return self.label






