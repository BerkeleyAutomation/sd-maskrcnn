import json, datetime

from django.http import HttpResponse, JsonResponse
from django.views.decorators.cache import never_cache
from django.views import View
from django.utils.decorators import method_decorator

from django.conf import settings

from . import models


class LabellingToolView (View):
    """
    Labelling tool class based view

    Subclass and override the `get_labels` method (mandatory) and optionally
    the `update_labels` method to customise how label data is accessed and updated.

    `get_labels` should return either a `models.Labels` instance or a dictionary of the form:
    `{
        'complete': boolean indicating if labelling is finished for this image,
        'labels': label data as JSON
        'state': [optional] 'editable' if editing should be permitted, 'locked' if the UI should
            warn the user that the labels are being edited by someone else
    }`

    Example:
    >>> class MyLabelView (LabellingToolView):
    ...     def get_labels(self, request, image_id_str, *args, **kwargs):
    ...         image = models.Image.get(id=int(image_id_string))
    ...         # Assume `image.labels` is a field that refers to the `Labels` instance
    ...         return image.labels

    Or:
    >>> class MyLabelView (LabellingToolView):
    ...     def get_labels(self, request, image_id_str, *args, **kwargs):
    ...         image = models.Image.get(id=int(image_id_string))
    ...         # Lets assume that the label data has been incorporated into the `Image` class:
    ...         labels_metadata = {
    ...             'complete': image.complete,
    ...             'timeElapsed': image.edit_time_elapsed,
    ...             'labels': image.labels_json,
    ...             'state': ('locked' if image.in_use else 'editable')
    ...         }
    ...         return labels_metadata
    ...
    ...     def update_labels(self, request, image_id_str, labels, complete, time_elapsed, *args, **kwargs):
    ...         image = models.Image.get(id=int(image_id_string))
    ...         image.complete = complete
    ...         image.edit_time_elapsed = time_elapsed
    ...         image.labels_json = labels
    ...         image.save()
    """
    def get_labels(self, request, image_id_str, *args, **kwargs):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def update_labels(self, request, image_id_str, labels, complete, time_elapsed, *args, **kwargs):
        labels = self.get_labels(request, image_id_str, *args, **kwargs)
        labels.update_labels(labels, complete, time_elapsed, request.user, save=True, check_lock=False)

    @method_decorator(never_cache)
    def get(self, request, *args, **kwargs):
        if 'labels_for_image_id' in request.GET:
            image_id_str = request.GET['labels_for_image_id']

            labels = self.get_labels(request, image_id_str, *args, **kwargs)
            if labels is None:
                # No labels for this image
                labels_header = {
                    'image_id': image_id_str,
                    'complete': False,
                    'timeElapsed': 0.0,
                    'state': 'editable',
                    'labels': [],
                }
            elif isinstance(labels, models.Labels):
                # Remove existing lock
                labels_header = {
                    'image_id': image_id_str,
                    'complete': labels.complete,
                    'timeElapsed': labels.edit_time_elapsed,
                    'state': 'editable',
                    'labels': labels.labels_json,
                }
            elif isinstance(labels, dict):
                labels_header = {
                    'image_id': image_id_str,
                    'complete': labels['complete'],
                    'timeElapsed': labels.get('edit_time_elapsed', 0.0),
                    'state': labels.get('state', 'editable'),
                    'labels': labels['labels'],
                }
            else:
                raise TypeError('labels returned by get_labels metod should be None, a Labels model '
                                'or a dictionary; not a {}'.format(type(labels)))

            return JsonResponse(labels_header)
        elif 'next_unlocked_image_id_after' in request.GET:
            return JsonResponse({'error': 'operation_not_supported'})
        else:
            return JsonResponse({'error': 'unknown_operation'})

    def post(self, request, *args, **kwargs):
        labels = json.loads(request.POST['labels'])
        image_id = labels['image_id']
        complete = labels['complete']
        time_elapsed = labels['timeElapsed']
        label_data = labels['labels']

        try:
            self.update_labels(request, str(image_id), label_data, complete, time_elapsed, *args, **kwargs)
        except models.LabelsLockedError:
            return JsonResponse({'error': 'locked'})
        else:
            return JsonResponse({'response': 'success'})



class LabellingToolViewWithLocking (LabellingToolView):
    """
    Labelling tool class based view with label locking

    Subclass and override the `get_labels` method (mandatory), the
    `get_next_unlocked_image_id_after` method (mandatory) and optionally
    the `update_labels` method to customise how label data is accessed and updated.

    `get_labels` should return a `models.Labels` instance; it should NOT return anything else
    in the way that the `get_labels` method of a subclass of `LabellingToolView` can.

    The `LABELLING_TOOL_LOCK_TIME` attribute in settings can be used to set the amount of time
    that a lock lasts for in seconds; default is 10 minutes (600s).

    Example:
    >>> class MyLabelView (LabellingToolViewWithLocking):
    ...     def get_labels(self, request, image_id_str, *args, **kwargs):
    ...         image = models.Image.get(id=int(image_id_string))
    ...         # Assume `image.labels` is a field that refers to the `Labels` instance
    ...         return image.labels
    ...
    ...     def get_next_unlocked_image_id_after(self, request, current_image_id_str, *args, **kwargs):
    ...         unlocked_labels = image_labelling_tool.models.Labels.objects.unlocked()
    ...         unlocked_imgs = models.Image.objects.filter(labels__in=unlocked_labels)
    ...         unlocked_img_ids = [img.id for img in unlocked_imgs]
    ...         try:
    ...             index = unlocked_img_ids.index(int(current_image_id_str))
    ...         except ValueError:
    ...             return None
    ...         index += 1
    ...         if index < len(unlocked_img_ids):
    ...             return unlocked_img_ids[index]
    ...         else:
    ...             return None
    """
    def get_next_unlocked_image_id_after(self, request, current_image_id_str, *args, **kwargs):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def update_labels(self, request, image_id_str, labels_js, complete, time_elapsed, *args, **kwargs):
        expire_after = getattr(settings, 'LABELLING_TOOL_LOCK_TIME', 600)
        labels = self.get_labels(request, image_id_str, *args, **kwargs)
        labels.update_labels(labels_js, complete, time_elapsed, request.user, check_lock=True, save=False)
        if request.user.is_authenticated():
            labels.refresh_lock(request.user, datetime.timedelta(seconds=expire_after), save=False)
        labels.save()

    @method_decorator(never_cache)
    def get(self, request, *args, **kwargs):
        if 'labels_for_image_id' in request.GET:
            image_id_str = request.GET['labels_for_image_id']

            labels = self.get_labels(request, image_id_str)

            if not isinstance(labels, models.Labels):
                raise TypeError('labels returned by get_labels metod should be a Labels '
                                'model, not a {}'.format(type(labels)))

            # Remove existing lock
            if request.user.is_authenticated():
                already_locked = models.Labels.objects.locked_by_user(request.user)
                for locked_labels in already_locked:
                    locked_labels.unlock(from_user=request.user, save=True)

            if labels.is_locked_to(request.user):
                state = 'locked'
                attempt_lock = False
            else:
                state = 'editable'
                attempt_lock = True
            labels_header = {
                'image_id': image_id_str,
                'complete': labels.complete,
                'timeElapsed': labels.edit_time_elapsed,
                'state': state,
                'labels': labels.labels_json,
            }

            if attempt_lock and request.user.is_authenticated():
                expire_after = getattr(settings, 'LABELLING_TOOL_LOCK_TIME', 600)
                labels.lock(request.user, datetime.timedelta(seconds=expire_after), save=True)

            return JsonResponse(labels_header)
        elif 'next_unlocked_image_id_after' in request.GET:
            current_image_id_str = request.GET['next_unlocked_image_id_after']
            next_image_id = self.get_next_unlocked_image_id_after(
                request, current_image_id_str, *args, **kwargs)

            return JsonResponse({'next_unlocked_image_id': str(next_image_id)})
        else:
            return JsonResponse({'error': 'unknown_operation'})
