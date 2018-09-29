"""ImageAnnotation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.conf.urls.static import static
from . import settings
from Annotation import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    # Views html
    url(r'^$', views.index, name='index'),
    url(r'^index/$', views.index, name='index'),
    # url(r'^index123/$', views.index123, name='index123'),
    # url(r'^origin/dashboard/$', views.dashboard, name='dashboard'),
    # url(r'^origin/grids/$', views.grids, name='grids'),
    # url(r'^origin/media/$', views.media, name='media'),
    # url(r'^origin/general/$', views.general, name='general'),
    # url(r'^origin/typography/$', views.typography, name='typography'),
    # url(r'^origin/widgets/$', views.widgets, name='widgets'),
    # url(r'^origin/inbox/$', views.inbox, name='inbox'),
    # url(r'^origin/compose/$', views.compose, name='compose'),
    # url(r'^origin/tables/$', views.tables, name='tables'),
    # url(r'^origin/forms/$', views.forms, name='forms'),
    # url(r'^origin/validation/$', views.validation, name='validation'),
    # url(r'^origin/login/$', views.login, name='login'),
    # url(r'^origin/signup/$', views.signup, name='signup'),
    # url(r'^origin/blank_page/$', views.blank_page, name='blank-page'),
    # url(r'^origin/charts/$', views.charts, name='charts'),

    url(r'^gallery$', views.gallery2, name='gallery_no_params'),
    url(r'^gallery/(\w+)$', views.gallery, name='gallery'),
    url(r'^slider/$', views.slider, name='slider'),

    url(r'^slider/add$', views.process_classify_1_by_2, name='process_classify_1_by_2'),
    url(r'^overview/$', views.overview, name='overview'),
    url(r'^upload/(\w+)/$', views.upload, name='upload'),
    url(r'^classify/$', views.classify, name='classify'),
    # url(r'^train/$', views.train, name='train'),
    # url(r'^validate/$', views.validate, name='validate'),
     url(r'^search/$', views.search, name='search'),

    # img process
    url(r'^update/label/$', views.update_image_label, name='update_label'),
    url(r'^upload/single/process/$', views.get_input_file_single, name='upload_single'),
    url(r'^upload/multiple/process/$', views.get_input_file_multiple, name='upload_multiple'),
    # url(r'^delete/$', views.delete_image, name="delete_image"),

    # Classify
    url(r'^upload/classify/images/$', views.get_classify_image, name="upload_classify"),
    url(r"^classify/process/$", views.process_classify, name="process_classify"),
    # url(r"^classify/process/$", views.classify_result, name="process_classify"), # 跳转到新页面，已放弃
    url(r'^classify/process/1_by_1$', views.process_classify_1_by_1, name="process_classify_1_by_1"),

    # Train Process
    # url(r'^train/process/$', views.process_train, name="process_train"),

    # Validate
    # url(r"^validate/process/$", views.process_validate, name="process_validate"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
