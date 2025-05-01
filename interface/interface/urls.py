"""
URL configuration for interface project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import RedirectView
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Redirection de la racine vers login
    path('', RedirectView.as_view(url='login/')),

                  # Applications
    path('', include('accounts.urls')),  # URLs pour l'authentification
    path('pointscloud_upload/', include('pointscloud_upload.urls')),  # Upload et classification




] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

