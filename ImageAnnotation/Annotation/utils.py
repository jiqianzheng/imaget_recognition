import os
import os.path as osp
from django.conf import settings
import re

import numpy as np
from Annotation import models
import random

labels = None

files = None

key_value={'shu':"树",
           'shantou':"山头",
           'shanpo':"山坡",
           'fangwu':"房屋",
           'stzh':"山头褶",
           'fantou':"矾头",
            'tikuan':"题款",
           'yinzhang':"印章",
           'qiao':"桥",
           'chengguan':"城关"
}

# def get_all_labels():
#     global labels
#     if labels is None:
#         #获取所有ImageDB路径
#         labels = os.listdir(osp.join(settings.BASE_DIR, "ImagesDB"))
#         print(labels)
#         #获取路径的名称以及对应每个文件下的文件的个数
#         labels = [{"label": label, "num": len(os.listdir(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label)))} for label in labels]
#     return labels
def  get_all_labels():
    global labels
    if labels is None:
        labels=models.Category.objects.all()
        labels = [{"label": key_value[label.label], "num":m.num } for label in labels for m in models.Category.objects.all().filter(label=label)]
    return  labels
def  get_image_labels():
    labels=None
    if labels is None:
        labels=models.Painting.objects.all()
        print("get_image_labels labels=%s"%(labels))
    return  labels
def get_all_images():
    global files
    if files is None:
        # labels = [item["label"] for item in get_all_labels()]
        labels=["shan","shu"]
        files = []
        prefix = osp.join(settings.BASE_DIR, "ImagesDB")
        for label in labels:
            #获取图片类文件标签下所有的图片名称
            x = os.listdir(osp.join(prefix, label))
            #此处的static代表的是web根目录，从定义的ImageDB中查找
            x = list(map(lambda item: "".join([r"/static/", label, "/", item]), x))
            # 纵向合并，类似于append
            files = np.hstack((files, x))
    return files

def get_random_pating():
    # global files_image
    # global  file_images_all
    # if files_image is None:
    #     labels=models.Painting.objects.all()
    #
    #     file_images_all={}
    #     if labels.exists():
    #        x = labels.count()
    #        for p in range(9):
    #           files_image = {}
    #           id= random.randint(1,x)
    #           obj=models.Psection.objects.all().filter(pid=id)
    #
    #           # print(p,id)
    #           # print(labels)
    #           # print(labels[id-1])
    #           y=labels[id-1]
    #           files_image[p]=y
    #           file_images_all[obj]=files_image
    #     print(file_images_all)
    # print("获取随机图片完成！！！！！！！！")
    # return file_images_all
    # return files_image
    pass


# 图片存入目录
def write2disk(file_path,data_path, file_data):
    isExits = os.path.exists(file_path)
    if not isExits:
        os.makedirs(file_path)

    file_data.save(data_path)



def resolve_file_name(file_path):
    a = file_path.split("/")
    return a[0], a[1]


def resolve_report(report_string):
    # print(report_string)
    rows = report_string.split("\n")
    r = [re.split(r"\s+", rows[0], re.S)[-5:-1]]
    for row in rows[2:-3]:
        r.append(re.split(r"\s+", row, re.S)[-5:-1])
    r.append(re.split(r"\s+", rows[-2], re.S)[-5:-1])
    r[-1][0] = "avg/total"
    return r

