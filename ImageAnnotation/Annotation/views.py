# coding=utf-8
import os
import os.path as osp
import shutil
from PIL import ImageDraw
from extract_cnn_vgg16_keras import extract_feat
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from django.views.decorators.csrf import csrf_exempt

from Annotation import  models
from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from django.conf import settings
from django.template import RequestContext
from django.template.loader import render_to_string
from .forms import SearchForm
from .utils import get_all_labels
from .utils import write2disk
from .utils import get_random_pating
from .utils import get_all_images
import tensorflow as tf
import matplotlib.image as mpimg
from nets import  np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from PIL import Image
import  random
import sys
sys.path.append('../')




# '''
slim = tf.contrib.slim
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)


reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
# 调用模型的路径
ckpt_filename = './checkpoints/model.ckpt-91516'
print(ckpt_filename)
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.2, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores,rbboxes

VOC_LABELS = {
    'no'
    'ne': (0, 'Background'),
    'chengguan': (1, "ChengGuan"),
    'fangwu': (2, "FangWu"),
    'fantou': (3, "FanTou"),
    'qiao': (4, "Qiao"),
    'shanpo': (5, "ShanPo"),
    'shantou': (6, "ShanTou-h"),

    'shu': (7, "Shu-cy"),
    'stzh': (8, "STZH-c"),
    'tikuan': (9, "TiKuan-qc"),
    'yinzhang': (10, "YinZhang-qc"),
}

value_key={"树":'shu',
           "山头":'shantou',
           "山坡":'shanpo',
           "房屋":'fangwu',
           "山头褶":'stzh',
           "矾头":'fantou',
            "题款":'tikuan',
           "印章":'yinzhang',
           "桥":'qiao',
           "城关":'chengguan'
}

labels=list(VOC_LABELS.keys())
# '''
# 主页面的方法
def index(request):
    return render(request, 'index.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
# 暂时没用到
def transfer(request):
    return render(request, 'transfer.html')
# 瀑布流用到的方法
def gallery(request, label):
    # file_list = os.listdir(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label))
    # file_list = list(map(lambda item: r"/static/" + label + r"/" + item, file_list))
    label = value_key[label]
    print('label',label)
    file_list = []
    obj = models.Psection.objects.filter(label=label)
    for b in obj:
        # print(b.label)
        file_list.append(r"/" + str(b.section_url))
    print(file_list)
    print(get_all_labels())
    return render(request, 'gallery.html', {"File": file_list, "Labels": get_all_labels(), "SearchForm": SearchForm()})
# 暂时没用
def gallery2(request):
    return render(request, 'index.html', {"File": get_all_labels(), "SearchForm": SearchForm()})

# 这里是轮播图展示页面调用的方法
def slider(request):
        list1= []
        list2= []
        list3= []
        list4 =[]
        list5= []
        list6= []
        list7= []
        list8= []
        list9= []
    # if files_image is None:
        files_image = {}
        labels = models.Painting.objects.all()
        if labels.exists():
            x = labels.count()
            print("图片的数量 %s"%(x))
            # 从数据库中获取下面id的图片
            id=[85,103,84,55,62,69,47,49,107]
            for p in range(9):
                # list1=[]
                # 下面是和上面固定图片不同，随机获取9张图片，不再是固定的图片
                # id = random.randint(1, x)
                # print(id)
                obj = models.Psection.objects.filter(pid=id[p])
                print("标签数%s"%(obj.count()))
                # print(p)
                if p == 0:
                    for b in obj:
                        # print(b.label)
                        list1.append({b.label:"/" + str(b.section_url)})
                # print(list1)
                elif p == 1:
                    for b in obj:
                        list2.append({b.label: "/" + str(b.section_url)})
                elif p == 2:
                    for b in obj:
                        list3.append({b.label: "/" + str(b.section_url)})
                elif p == 3:
                    for b in obj:
                        list4.append({b.label: "/" + str(b.section_url)})
                elif p == 4:
                    for b in obj:
                        list5.append({b.label: "/" + str(b.section_url)})
                elif p == 5:
                    for b in obj:
                        list6.append({b.label: "/" + str(b.section_url)})
                elif p == 6:
                    for b in obj:
                        list7.append({b.label: "/" + str(b.section_url)})
                elif p == 7:
                    for b in obj:
                        list8.append({b.label: "/" + str(b.section_url)})
                elif p == 8:
                    for b in obj:
                        list9.append({b.label: "/" + str(b.section_url)})
                # print("%%%%%%%%%%%%%")
                # print(id[p])
                y = labels[id[p]- 1]
                # print(y)
                # print("%%%%%%%%%%%%%")
                files_image[p] = y
        print("获取随机图片完成！！！！！！！！")
        # print((list1))
        # print((list2))
        # print((list3))
        # print((list4))
        # print((list5))
        # print((list6))
        # print((list7))
        # print((list8))
        print((list9))
        return render(request, 'slider.html', {"Labels": get_all_labels(), "File": files_image,"list1":list1,"list2":list2,"list3":list3,"list4":list4,"list5":list5,"list6":list6,"list7":list7,"list8":list8,"list9":list9,"SearchForm": SearchForm()})

#
@csrf_exempt
def process_classify_1_by_2(request):

    # 获取页面传来的数据
    if request.method == "POST":
        data=request.POST.get("image")
        print(data)
        data=data[1:]
        label=str(request.POST.get("label"))
        # print(label)
        # print(data)

        # 相似性查询，调用训练的相似性模型来查找相似的图片
        h5f = h5py.File('quanshanshui_index6688', 'r')
        feats = h5f['dataset_1'][:]
        imgNames = h5f['dataset_2'][:]
        h5f.close()

        queryDir = data
        queryVec = extract_feat(queryDir)
        scores = np.dot(queryVec, feats.T)
        rank_ID = np.argsort(scores)[::-1]

        maxres = 20
        i = 0

        imlist1 = [imgNames[index] for i, index in enumerate(rank_ID[1:maxres])]
        print("*****************************************")
        print(imlist1)
        images={}
        yuantu={}


        result=[]
        i=0
        # 查找到相似图，并计算出框在原图的位置
        for j in imlist1:
            j=str(j)
            j=j[2:-5]
            print(j)
            obj=models.Psection.objects.filter(sectionname=str(j))
            if obj.exists():
                # print("**********")
                str1 ="/"+str(obj[0].section_url)
                label2=str(obj[0].label)
                # print(str1)
                # print(label2)
                # print(label)
                if label == label2:
                   images[i] = str1

                   obj2=models.Painting.objects.filter(id=obj[0].pid)
                   gg="/media/"+str(obj2[0].picture_url)
                   filename=obj2[0].filename
                   print("filenme",filename)
                   border_list=[]

                   yu_width = obj2[0].width
                   yu_height=obj2[0].height

                   xmin=obj[0].xmin
                   ymin=obj[0].ymin
                   xmax=obj[0].xmax
                   ymax=obj[0].ymax

                   border_width=xmax-xmin
                   border_height=ymax-ymin

                   # print("______________")
                   # print(obj2[0].id)
                   # print(filename)
                   # print("图的宽度 %s"%yu_width)
                   # print("图的高度 %s"%yu_height)
                   # print("$$$$$$$$$")
                   # print("xmin %s"%xmin)
                   # print("xmax %s"%xmax)
                   # print("边框的宽度%s"%border_width)
                   # print("ymin %s"%ymin)
                   # print("ymax %s" %ymax)
                   # print("边框的高度%s"%border_height)
                   # print("$$$$$$$$$")

                   h_w_rate=1
                   w_h_rate=1
                   h_rate = 1
                   w_rate = 1
                   if yu_width>yu_height:
                       w_h_rate=yu_width/yu_height
                       print("w_h_rate %s"%w_h_rate)
                       if yu_width > 590:
                           w_rate = yu_width / 590;
                           h_rate=yu_height/(590/w_h_rate)
                           # print("yu_width > 590:")
                           # print("w_rate%s"%w_rate)
                           # print("h_rate%s"%h_rate)
                       elif yu_height > 700:
                           # print("yu_height > 700:")
                           # print("w_rate %s" %w_rate)
                           # print("h_rate%s" %h_rate)
                           h_rate = yu_height / 700
                           w_rate=yu_width/(700/w_h_rate)
                   else:
                       h_w_rate=yu_height/yu_width
                       print("h_w_rate %s"%h_w_rate)
                       if yu_height > 700:
                           h_rate = yu_height / 700
                           w_rate=yu_width/(700/h_w_rate)
                           # print("yu_height > 700:")
                           # print("w_rate%s" % w_rate)
                           # print("h_rate%s" % h_rate)
                       elif yu_width > 590:
                           w_rate = yu_width / 590;
                           h_rate=yu_height/(590/h_w_rate)
                           # print("yu_width > 590:")
                           # print("w_rate%s" % w_rate)
                           # print("h_rate%s" % h_rate)



                   # print(h_rate)
                   # print(w_rate)
                   border_height=border_height/h_rate
                   border_width=border_width/w_rate
                   xmin=xmin/w_rate
                   ymin=ymin/h_rate

                   # print("xmin %s" % xmin)
                   # print("ymin %s" % ymin)
                   # print("边框的宽度%s" % border_width)
                   # print("边框的高度%s" % border_height)
                   # print("______________")
                   border_list.append(xmin)
                   border_list.append(ymin+55)
                   border_list.append(border_width)
                   border_list.append(border_height)

                   border_list.append(filename)
                   border_list.append(gg)

                   yuantu[i] = border_list

                   i+=1

        # images[0]='/media/tikuan/tikuan13001203.jpg'
        # images[1]='/media/yinzhang/yinzhang1513432.jpg'
        # images[2]='/media/fantou/fantou12005074.jpg'
        print("相似查询")
        print(images)
        print(yuantu)
        print("相似查询")
        content = render_to_string("divtwo.html",
                                   {"image123": images,"yuantu":yuantu})
        result.append(content)
    return HttpResponse("".join(result))


def overview(request):
    return render(request, 'overview.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


def upload(request, is_single):
    if is_single == "single":
        return render(request, 'upload_s.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
    else:
        return render(request, 'upload_m.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})

# 这里是上传图片识别的页面
def classify(request):
    return render(request, "classify.html", {"Labels": get_all_labels(), "SearchForm": SearchForm()})

# 暂时也没用到
def classify_result(request):
    return render(request, "classify.html", {"Labels": get_all_labels(), "SearchForm": SearchForm()})
    # result = process_classify()
    # return render(request, "results_cls.html", {"Labels": get_all_labels(), "Results": result})

# def train(request):
#     form = TrainForm()
#     return render(request, "train.html", {"Labels": get_all_labels(), "form": form, "SearchForm": SearchForm()})
#
#
# def validate(request):
#     return render(request, "validate.html", {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
# def dashboard(request):
#     return render(request, 'dashboard.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


# def grids(request):
#     return render(request, 'grids.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


# def media(request):
#     return render(request, 'media.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def general(request):
#     return render(request, 'general.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def typography(request):
#     return render(request, 'typography.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def widgets(request):
#     return render(request, 'widgets.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def inbox(request):
#     return render(request, 'inbox.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def compose(request):
#     return render(request, 'compose.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def tables(request):
#     return render(request, 'tables.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def forms(request):
#     return render(request, 'forms.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def validation(request):
#     return render(request, 'validation.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def login(request):
#     return render(request, 'login.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def signup(request):
#     return render(request, 'signup.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def blank_page(request):
#     return render(request, 'blank-page.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})
#
#
# def charts(request):
#     return render(request, 'charts.html', {"Labels": get_all_labels(), "SearchForm": SearchForm()})


# 系统处理逻辑
def search(request):
    response = {"status": 0}
    print(request.POST)
    form = SearchForm(request.POST)
    if form.is_valid():
        response["label"] = request.POST["keyword"]
    else:
        response["status"] = -1
    return JsonResponse(response)


def update_image_label(request):
    """
    更新图片类别
    :param request: {oldURL: "战舰/zhanjian_01.jpg", newLabel: "坦克"}
    :return:
    """
    old_url = request.POST.get("oldURL").split("/")
    filename = old_url[1]
    old_label = old_url[0]
    new_label = request.POST.get("newLabel")
    old_url = osp.join(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), old_label), filename)
    new_url = osp.join(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), new_label), filename)
    if osp.exists(new_url):
        new_url = osp.splitext(new_url)[0] + ".1" + osp.splitext(new_url)[1]
    os.rename(old_url, new_url)
    response = {"status": 0}
    return JsonResponse(response)


def get_input_file_single(request):
    """
    获取上传的图片，该方法为按类别上传的处理逻辑
    :param request:
    :return:
    """
    response = {"status": 0}
    if request.method == "POST":
        label = request.POST.get("label")
        file_data = request.FILES.get(r"image_file", "没有图片")
        filename = file_data.name
        file_path = osp.join(osp.join(osp.join("/home/htc/Documents", "test"), label), filename)
        print(file_data)
        obj= models.Painting(picture_url=file_data)
        obj.save()
        # write2disk(file_path, file_data)
    return JsonResponse(response)


def get_input_file_multiple(request):
    """
    接收上传的.zip文件，保证文件类型正确
    :param request:
    :return:
    """
    response = {"status": 0}
    if request.method == "POST":
        file_data = request.FILES.get(r"image_file_zip", "没有数据")
        file_name = file_data.name
        file_path = osp.join(osp.join("/home/htc/Documents", "test"), file_name)
        write2disk(file_path, file_data)
        shutil.unpack_archive(file_path, osp.splitext(file_path)[0])
    return JsonResponse(response)

#
# def delete_image(request):
#     """
#     删除图片
#     :param request:
#     :return:
#     """
#
#     response = {"status": 0}
#     label, filename = resolve_file_name(request.POST.get("file_path"))
#     file_path = osp.join(osp.join(osp.join(settings.BASE_DIR, "ImagesDB"), label), filename)
#     print(file_path)
#     # os.remove(file_path)
#     return JsonResponse(response)

# 这里是ajax上传图片进行ssd识别
def get_classify_image(request):
    """
    保存待标注的上传图像
     """
    # pass
    response = {"status": 0}
    if request.method == "POST":
        file_data= request.FILES.get(r"image", "没有图片")
        pending=str(file_data.name)[-4:]
        img = mpimg.imread(file_data)
        image = Image.open(file_data)
        # image=Image.open(file_data)
        rclasses, rscores, rbboxes = process_image(img)
        # print('@@@@@@@')
        height = img.shape[0]
        width = img.shape[1]
        pid1 = models.Painting.objects.all()

        if pid1.exists():
            pid=pid1.order_by('-id')[0].id
        else :
            pid=0
        # 看有多少行
        # print(pid)
        id_list =[]
        # loaction={}
        colors = dict()
        colors[1] = (84 / 256, 255 / 256, 159 / 256)
        colors[2] = (100 / 256, 149 / 256, 237 / 256)
        colors[3] = (255 / 256, 255 / 256, 0 / 256)
        colors[4] = (255 / 256, 106 / 256, 106 / 256)
        colors[5] = (255 / 256, 105 / 256, 180 / 256)
        colors[6] = (255 / 256, 48 / 256, 48 / 256)
        colors[7] = (160 / 256, 32 / 256, 240 / 256)
        colors[8] = (255 / 256, 165 / 256, 0 / 256)
        colors[9] = (0 / 256, 255 / 256, 0 / 256)
        colors[10] = (0 / 256, 245 / 256, 255 / 256)
        pating_data_Url = None
        # plt.figure(figsize=(width, height))
        plt.imshow(img)

        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id >= 0:
                score = rscores[i]
                ymin = int(rbboxes[i, 0] * height)
                xmin = int(rbboxes[i, 1] * width)
                ymax = int(rbboxes[i, 2] * height)
                xmax = int(rbboxes[i, 3] * width)
                # loaction[cls_id]=(xmin,ymin,xmax,ymax)
                class_name = labels[cls_id]
                region = (int(xmin), int(ymin), int(xmax), int(ymax))
                cropImg = image.crop(region)
                file_data_name=class_name+str(random.randint(1,100000000))
                file_path=osp.join("media", class_name)
                file_data_Url=file_path+'/'+file_data_name+pending
                # print("+++++++++")
                # print(class_name,file_data)
                obj = models.Psection(pid=pid+1, label=class_name,sectionname=file_data_name,xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,score=score
                                      ,section_url=file_data_Url)
                obj.save()
                write2disk(file_path,file_data_Url,cropImg )
                pp=models.Category.objects.filter(label=class_name)
                if pp.exists():
                    num = pp[0].num+1
                    # print(num)
                    pp.update(num=num)
                else:
                    num = 1
                    obj=models.Category(label=class_name,num=num)
                    obj.save()
                # print("++++++")
                id = models.Psection.objects.all().order_by('-id')[0]
                id_list.append(id.id)
                # print("图片剪切完成")

                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=1.5)
                plt.gca().add_patch(rect)
                pating_url = osp.join("media",'pating_img' )
                # print(pating_url)
                pating_data_Url ='pating_img' + '/' + file_data.name

                isExits = os.path.exists(pating_url)
                if not isExits:
                    os.makedirs(pating_url)
                print("构造目录完成")
                # plt.figure(figsize=(width,height))

                # plt.show()
                # plt.close(0)


            # print("原图画框完成")

        # for idx,item in loaction.items():
        #     print(item)
        #     draw = ImageDraw.Draw(image)
        #     draw.rectangle((int(xmin), int(ymin), int(xmax), int(ymax)), outline='blue')
        #     pating_url= file_path=osp.join("media",file_data.name)
        #     pating_data_Url=pating_url+'/'+file_data_name+pending
        #     write2disk(file_path, pating_data_Url, image)
        #     print("原图画框完成")
        # frame = plt.gca()
        # # y 轴不可见
        # frame.axes.get_yaxis().set_visible(False)
        # # x 轴不可见
        # frame.axes.get_xaxis().set_visible(False)
        plt.savefig("media/" + pating_data_Url)
        plt.clf()
        plt.close()
        obj = models.Painting(width=width, height=height, id_list=id_list, filename=file_data.name,
                              picture_url=file_data,pating_url=pating_data_Url)
        obj.save()
        print("图像上传自动处理完成")
    return JsonResponse(response)


def process_classify(request):

    pass
# 当点击开始标注的时候展示标注结果，标注过程在上传就处理了
def process_classify_1_by_1(request):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    pp = models.Painting.objects.all()
    if pp.exists():
        pid =pp.order_by('-id')[0].id
    else:
        pid = 0
    pp=models.Psection.objects.all().filter(pid=pid)
    images = []
    labels = []
    probs = []
    result = []
    for p in pp:
        score=float(p.score)*100.0
        score="%.4f" % score
        images.append(p.label)
        str1="/"+str(p.section_url)
        # str2="/"+str1
        labels.append(str1)
        probs.append({p.label:score})
        # print(str2)
    for idx, name in enumerate(images):

        # print(labels[idx])
        content = render_to_string("results_cls_v2.html",
                                   {"name": name, "path":labels[idx], "probs": probs[idx]})
        result.append(content)
    return HttpResponse("".join(result))
     # pass