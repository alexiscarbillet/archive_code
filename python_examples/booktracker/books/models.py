from django.db import models
from django.contrib.auth.models import User

class Book(models.Model):
    STATUS_CHOICES = [
        ('WTR', 'Want to Read'),
        ('RDG', 'Reading'),
        ('FIN', 'Finished'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=3, choices=STATUS_CHOICES, default='WTR')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
