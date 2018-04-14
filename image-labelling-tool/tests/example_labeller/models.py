from django.db import models

from image_labelling_tool import models as lt_models

# Create your models here.
class ImageWithLabels (models.Model):
    # image
    image = models.ImageField(blank=True)

    # labels
    labels = models.OneToOneField(lt_models.Labels, related_name='image')
