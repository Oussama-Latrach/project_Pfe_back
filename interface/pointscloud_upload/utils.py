import os
from django.conf import settings

def clean_old_files():
    """Nettoyer les fichiers de résultats anciens"""
    results_dir = os.path.join(settings.MEDIA_ROOT, 'classification_results')
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")