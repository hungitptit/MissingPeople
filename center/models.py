# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove ` ` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(blank=True, max_length=20)
    address = models.CharField(blank=True, max_length=150)
    city = models.CharField(blank=True, max_length=20)
    country = models.CharField(blank=True, max_length=50)
    image = models.ImageField(blank=True, upload_to='images/users/')

    def __str__(self):
        return self.user.username

    def user_name(self):
        return self.user.first_name + ' ' + self.user.last_name + ' [' + self.user.username + '] '

    def image_tag(self):
        return mark_safe('<img src="{}" height="50"/>'.format(self.image.url))
    image_tag.short_description = 'Image'
    class Meta:
         
        db_table = 'UserProfile'

class MissingPeople(models.Model):
    id = models.AutoField(db_column='ID', primary_key=True)  # Field name made lowercase.
    user = models.ForeignKey(User,models.DO_NOTHING, db_column='UserID')
    name = models.CharField(db_column='name', max_length=255, null=True, blank=True)
    gender= models.CharField (db_column='gender', max_length=100, null=True, blank=True)
    status = models.CharField(db_column='status',max_length=255, null=True,  blank=True)
    image = models.ImageField(db_column='image', null=True, upload_to='missing')
    description = models.CharField(db_column='description', blank=True, max_length=2550, null=True)
    location = models.CharField(db_column='location', max_length=255, blank=True, null=True)
    representation = models.BinaryField(db_column='representation', blank=True, null= True)
    date = models.DateField(db_column='date',blank=True, null= True)

    class Meta:
         
        db_table = 'MissingPeople'


'''''
class ReportedPeople(models.Model):
    id = models.AutoField(db_column='ID', primary_key=True)  # Field name made lowercase.
    user = models.ForeignKey(User,models.DO_NOTHING, db_column='UserID')
    name = models.CharField(db_column='name', max_length=255, null=True, blank=True)
    gender= models.CharField (db_column='gender', max_length=100, null=True, blank=True)
    status = models.CharField(db_column='status',max_length=255, null=True,  blank=True)
    image = models.ImageField(db_column='image', null=True, upload_to='reported')
    description = models.CharField(db_column='description', blank=True, max_length=2550, null=True) 
    location = models.CharField(db_column='location', max_length=255, blank=True, null=True) 
    representation = models.BinaryField(db_column='representation', blank=True, null= True)
    class Meta:
         
        db_table = 'ReportedPeople'
'''

class Address(models.Model):
    id = models.AutoField(db_column='ID', primary_key=True)  # Field name made lowercase.
    city = models.CharField(db_column='City', max_length=255, blank=True, null=True)  # Field name made lowercase.
    district = models.CharField(db_column='District', max_length=255, blank=True, null=True)  # Field name made lowercase.
    town = models.CharField(db_column='Town', max_length=255, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        
        db_table = 'address'

class MatchedPairs(models.Model):
   id = models.AutoField(db_column='ID', primary_key=True)  # Field name made lowercase.
   missingName = models.CharField(db_column='missing_name', max_length=255, null=True, blank=True)
   reportedName = models.CharField(db_column='reported_name', max_length=255, null=True, blank=True)
   missingImage = models.ImageField(db_column='missing_image', null=True, upload_to='matched')
   reportedImage = models.ImageField(db_column='reported_image', null=True, upload_to='matched')
   class Meta:
        
        db_table = 'MatchedPairs'

