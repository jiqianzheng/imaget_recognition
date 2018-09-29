import os
import django
import  re
from PIL import Image
from django.core.files.base import ContentFile

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ImageAnnotation.settings")
django.setup()
from Annotation import models
from django.core.files import File

# load_path="/Users/qianzheng/Downloads/fd3/"
# pathDir=os.listdir(load_path)
# for path_dir in pathDir:
#     if path_dir !='.DS_Store':
#         p=os.listdir(load_path+path_dir)
#         i=0
#         id_list=[]
#         # print(p)
#         for pp in p:
#             if pp[:-4]==path_dir:
#                 print(pp[:-4])
#             elif pp[-4:]=='.jpg':
#                 pass
width=100
height=100
id_list=[1,2,3]
filename="北山"

img=open('/Users/qianzheng/Downloads/fd/9.jpg','rb')
print(img)
myfile=File(img)

print(myfile)
# img.save()
obj=models.Painting(width=width,height=height,id_list=id_list,filename=filename,picture_url= 'img/5.jpg')
obj.save()