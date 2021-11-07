from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, request
from tensorflow.python.eager.context import context
from .forms import *
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import cv2
import numpy as np
from PIL import Image
import os
#from MissingPeople import settings
from . import facenet
import pickle
from functools import cmp_to_key
import base64
#from django.db.models import Q


'''''
def comparator(source_representation, first_person, second_person):

    first_representation = base64.b64decode(first_person.representation)
    first_representation = pickle.loads(first_representation)
    second_representation = base64.b64decode(second_person.representation)
    second_representation = pickle.loads(second_representation)
    first_distance =  facenet.find_cosine_distance(source_representation,first_representation)
    second_distance = facenet.find_cosine_distance(source_representation,second_representation)
    if first_distance < second_distance:
        return -1
    elif first_distance == second_distance:
        return 0
    else:
        return 1
'''''

def sort_missing_list(source_representation, missing_list):
    
    '''
    Sort a list of missing people by face similary
    Input: 
        source_representation: The representation of source image (numpy array)
        missing_list: The list of missing people that need sorting
    Output: 
        miss

    '''
    sorted_list = sorted(missing_list, key=lambda p: facenet.find_cosine_distance(source_representation,pickle.loads(base64.b64decode(p.representation))))
    return sorted_list

def auto_search(request,id):

    missing = MissingPeople.objects.filter(id=id)
    gender = missing[0].gender
    name = missing[0].name
    missing_list = MissingPeople.objects.filter(status='2',gender=gender)
    print(missing_list)
    representation = base64.b64decode(missing[0].representation)
    representation = pickle.loads(representation)
    #print(representation)
    if len(representation) == 0:
        messages.warning(request,"Không thể phát hiện khuôn mặt trong ảnh của bạn, vui lòng cập nhật ảnh chân dung rõ hơn")
        return HttpResponseRedirect('/missingdetail?missing_id='+str(id))
    missing_list = sort_missing_list(source_representation=representation, missing_list=missing_list)
    #user = User.objects.filter(id=missing[0].user.id)
    return render(request, 'auto_search.html', {'missing_list': missing_list, 'title': "Kết quả gợi ý cho " + name })
    
@login_required(login_url='/login') # Check login
def user_view(request):
    current_user = request.user
    #missing_list = MissingPeople.objects.filter(Q(user=current_user), Q(status ='0')| Q(status='1'))
    waiting_list = MissingPeople.objects.filter(user = current_user, status = '0')
    missing_list = MissingPeople.objects.filter(user = current_user, status = '1')
    reported_list = MissingPeople.objects.filter(user = current_user, status = '2')

    context = {
        'waiting_list': waiting_list,
        'missing_list': missing_list,
        'reported_list': reported_list
    }
    return render(request, 'user.html',context)
@login_required(login_url='/login') # Check login
def missing_image_view(request):
  
    if request.method == 'POST':
        form = MissingForm(request.POST, request.FILES)
        
        if form.is_valid():
            data = MissingPeople()
            current_user= request.user
            data.user=current_user
            data.name = form.cleaned_data['name']
            data.gender = form.cleaned_data['gender']
            data.description =form.cleaned_data['description']
            data.location = form.cleaned_data['location']
            data.status =form.cleaned_data['status']
            data.image = form.cleaned_data['image']
            x = int(form.cleaned_data['x'])
            y = int(form.cleaned_data['y'])
            w = int(form.cleaned_data['width'])
            h = int(form.cleaned_data['height'])
            face_box = {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
            #print(data.image.url)
            #img = cv2.imdecode(np.fromstring(data.image, np.uint8), cv2.IMREAD_UNCHANGED)
            img = Image.open(data.image).convert("RGB")
            #np_img = np.array(Image.open(data.image).convert("RGB"))
            #img = cv2.imread('D:/study/doan/MissingPeople'+data.image.url)
            #img = cv2.imread('..'+data.image.url)
            #np_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            #print(img)
            #classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
            represent = None
            if ( x is None and y is None and w is None and h is None):
                represent = facenet.represent_image(img, auto=True)
            else:
                face = img.crop((x, y, w+x, h+y))
                represent = facenet.represent_image(img=face, auto=False)
            np_bytes = pickle.dumps(represent)
            np_base64 = base64.b64encode(np_bytes)
            data.representation = np_base64
            #cv2.imshow("test", img)
            #cv2.waitKey(0)
            data.save()
            messages.success(request, 'Lưu thành công!')
            return HttpResponseRedirect('/upload')
    else:
        form = MissingForm()
    return render(request, 'add_missing_person.html', {'form' : form})
  
def display_missing_people(request):
  
    if request.method == 'GET':
  
        # getting all the objects of missing.
        list = MissingPeople.objects.all() 
        return render(request, 'missing_people.html',{'missing_images' : list})
def success(request):
    return HttpResponse('successfully uploaded')

def login_form(request):
    form = LoginForm(request.POST or None)
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            current_user =request.user
            print(current_user.id)
            userprofile=UserProfile.objects.get(user_id=current_user.id)
            request.session['userimage'] = userprofile.image.url
            # Redirect to a success page.
            messages.success(request,'Đăng nhập thành công với tài khoản '+str(current_user))
            return HttpResponseRedirect('/missing')
        else:
            messages.warning(request,"Đăng nhập không thành công !! Sai tên đăng nhập hoặc mật khẩu")
            return HttpResponseRedirect('/login')
    # Return an 'invalid login' error message.

    #category = Category.objects.all()
    context = {#'category': category
    'form' : form
     }
    return render(request, 'login_form.html',context)

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save() #completed sign up
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            # Create data in profile table for user
            current_user = request.user
            data=UserProfile()
            data.user_id=current_user.id
            data.image="images/users/user.png"
            data.save()
            messages.success(request, 'Đăng ký tài khoản thành công!')
            return HttpResponseRedirect('/login/')
        else:
            messages.warning(request,form.errors)
            return HttpResponseRedirect('/signup')


    form = SignUpForm()
    #category = Category.objects.all()
    context = {#'category': category,
               'form': form,
               }
    return render(request, 'signup_form.html', context)

def missing_list(request):

    missing_list = MissingPeople.objects.filter(status='1')

    if missing_list != None:
        return render(request, 'missing_list.html', {'items': missing_list, 'title':"Danh sách các trường hợp đang thất lạc"})
    else:
        return render(request, 'missing_list.html', {'items': []})

def reported_list(request):
    
    missing_list = MissingPeople.objects.all()
    if missing_list != None:
        return render(request, 'reported_list.html', {'items': missing_list, 'title':"Danh sách các trường hợp đã có người tìm thấy và báo cáo"})
    else:
        return render(request, 'reported_list.html', {'items': []})

def logout_func(request):
    logout(request)
    messages.success(request,'Đã đăng xuất')
    return HttpResponseRedirect('/login')
def index(request):
    return HttpResponse('success')

def missing_detail(request):
    missing = MissingPeople.objects.filter(id=request.GET.get('missing_id'))
    user = User.objects.filter(id=missing[0].user.id)
    gender = missing[0].gender
    missing_list = MissingPeople.objects.filter(status='2',gender=gender)
    representation = base64.b64decode(missing[0].representation)
    representation = pickle.loads(representation)
    #print(representation)
    if len(representation) == 0:
        messages.warning(request,"Không thể phát hiện khuôn mặt trong ảnh của bạn, vui lòng cập nhật ảnh chân dung rõ hơn")
        return HttpResponseRedirect('/missingdetail?missing_id='+str(id))
    missing_list = sort_missing_list(source_representation=representation, missing_list=missing_list)
    #user = User.objects.filter(id=missing[0].user.id)
    #return render(request, 'auto_search.html', {'missing_list': missing_list, 'missing_person':missing })
    return render(request, 'missing_detail.html', {'missing_list': missing_list[:6], 'missing': missing[0], 'user': user[0] })
 