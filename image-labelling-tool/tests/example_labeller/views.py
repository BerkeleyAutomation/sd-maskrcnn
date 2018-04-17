import os, datetime

from django.shortcuts import render, get_object_or_404

from django.conf import settings

from image_labelling_tool import labelling_tool
from image_labelling_tool import models as lt_models
from image_labelling_tool import labelling_tool_views

from . import models

def home(request):
    image_descriptors = [labelling_tool.image_descriptor(
            image_id=img.id, url=img.image.url,
            width=img.image.width, height=img.image.height) for img in models.ImageWithLabels.objects.all()]

    # Convert the label class tuples in `settings` to `labelling_tool.LabelClass` instances
    label_classes = [labelling_tool.LabelClass(*c) for c in settings.LABEL_CLASSES]

    context = {
        'label_classes': [c.to_json()   for c in label_classes],
        'image_descriptors': image_descriptors,
        'initial_image_index': 0,
        'labelling_tool_config': settings.LABELLING_TOOL_CONFIG,
    }
    return render(request, 'index.html', context)


class LabellingToolAPI (labelling_tool_views.LabellingToolViewWithLocking):
    def get_labels(self, request, image_id_str, *args, **kwargs):
        image = get_object_or_404(models.ImageWithLabels, id=image_id_str)
        return image.labels

    def get_next_unlocked_image_id_after(self, request, current_image_id_str, *args, **kwargs):
        unlocked_labels = lt_models.Labels.objects.unlocked()
        unlocked_imgs = models.ImageWithLabels.objects.filter(labels__in=unlocked_labels)
        unlocked_img_ids = [img.id for img in unlocked_imgs]
        try:
            index = unlocked_img_ids.index(int(current_image_id_str))
        except ValueError:
            return None
        index += 1
        if index < len(unlocked_img_ids):
            return unlocked_img_ids[index]
        else:
            return None
