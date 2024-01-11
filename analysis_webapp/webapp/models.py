from django.db import models

class Multimodal(models.Model):
    created = models.DateTimeField(auto_now=True)
    model_name = models.CharField(max_length = 100, blank = True, default = "")
    model_path = models.CharField(max_length = 100, blank = True, default = "")

    class Meta:
        ordering = ['created']

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        return super().save(force_insert, force_update, using, update_fields)

class Predictions(models.Model):
    multimodal = models.ForeignKey(Multimodal, on_delete=models.CASCADE)
    created = models.DateField(auto_now=True)
