from django import forms
from django.forms.fields import ImageField
from .models import *
from django.contrib.auth.forms import UserCreationForm, UserChangeForm

ADDRESS_CHOICES = ['Thành phố Hà Nội', 'Tỉnh Hà Giang', 'Tỉnh Cao Bằng', 'Tỉnh Bắc Kạn']
STATUS_CHOICES = [
    ('0','Lưu nháp'), 
    ('1','Đăng tìm')
]
GENDER_CHOICES = [
    ('Nam','Nam',), 
    ('Nữ','Nữ',), 
    ('Khác','Giới tính khác')]
class MissingForm(forms.ModelForm):
    name = forms.CharField(max_length=255, label="Tên")
    gender = forms.ChoiceField(
        label="Giới tính",
        required=False,
        
        choices=GENDER_CHOICES,
    )
    description = forms.CharField(max_length=2550,widget=forms.Textarea, label="Mô tả")

    status = forms.ChoiceField(
        required=False,
       
        choices=STATUS_CHOICES,
    )
    image = forms.ImageField( label="Ảnh chân dung")
    class Meta:
        model = MissingPeople
        fields = ['name', 'image', 'gender', 'status']

class ReportForm(forms.ModelForm):
    name = forms.CharField(max_length=255, label="Tên")
    gender = forms.ChoiceField(
        required=False,
        
        choices=GENDER_CHOICES,
    )
    description = forms.CharField(max_length=2550,widget=forms.Textarea)

    status = forms.ChoiceField(
        required=False,
       
        choices=STATUS_CHOICES,
    )
    image = models.ImageField()
    class Meta:
        model = MissingPeople
        fields = ['name', 'image', 'gender', 'status']


class SignUpForm(UserCreationForm):
    username = forms.CharField(max_length=30,label= 'User Name :')
    email = forms.EmailField(max_length=200,label= 'Email :')
    first_name = forms.CharField(max_length=100, help_text='First Name',label= 'First Name :')
    last_name = forms.CharField(max_length=100, help_text='Last Name',label= 'First Name :')

    class Meta:
        model = User
        fields = ('username', 'email','first_name','last_name', 'password1', 'password2', )

class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                            
                "class": "form-control"
            }
        ))
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                             
                "class": "form-control"
            }
        ))

class SignUpForm(UserCreationForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                            
                "class": "form-control"
            }
        ))
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                             
                "class": "form-control"
            }
        ))
    last_name = forms.CharField(
       
         widget=forms.TextInput(
            attrs={
                              
                "class": "form-control"
            }
        ))
    first_name = forms.CharField(
       
         widget=forms.TextInput(
            attrs={
                              
                "class": "form-control"
            }
        ))
   
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                    
                "class": "form-control"
            }
        ))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                             
                "class": "form-control"
            }
        ))

    class Meta:
        model = User
        fields = ('username', 'email','first_name','last_name', 'password1', 'password2' )