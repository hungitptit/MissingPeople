"""MissingPeople URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls.static import static
from django.urls.conf import include
from center.views import *
urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload', missing_image_view, name = 'upload'),
    path('success', success, name = 'success'),
    path('login/', login_form, name='login'),
    path('logout/', logout_func, name='logout'),
    path('signup/', signup, name='signup'),
    path('missing/', missing_list, name="missing"),
    path('', missing_list, name='home'),
    path('missing_people', display_missing_people, name = 'missing_people'),
    path('accounts/', include('allauth.urls')),
    path('missingdetail', missing_detail, name="productdetail"),
    

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)