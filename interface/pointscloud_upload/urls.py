from django.urls import path
from . import views
from .views import classify_pointcloud

urlpatterns = [
    path('api/classify/', classify_pointcloud, name='classify_pointcloud'),
    path('api/download/<str:download_id>/', views.download_classified_file, name='download_classified_file'),
]