from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, request
from .forms import *
from django.contrib import messages
from django.contrib.auth.decorators import login_required


# Create your views here.
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
            data.status =form.cleaned_data['status']
            data.image = form.cleaned_data['image']
            data.save()
            #form.save()
            return redirect('success')
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
    return render(request, 'missing_detail.html', {'missing': missing[0],'user': user[0] })
 
        