# accounts/models.py

from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Si vous souhaitez ajouter des champs supplémentaires à votre modèle d'utilisateur personnalisé, vous pouvez le faire ici.
    # Assurez-vous de ne pas redéfinir les champs qui sont déjà inclus dans AbstractUser, tels que `username` et `password`.
    pass

    class Meta:
        db_table = 'user_account'
