import json, datetime
from django.db import models
from django.conf import settings
from django.utils import timezone
from . import managers

class LabelsLockedError (Exception):
    pass

# Create your models here.
class Labels (models.Model):
    # Label data
    labels_json_str = models.TextField(default='[]')
    complete = models.BooleanField(default=False)

    # Creation date
    creation_date = models.DateField()

    # Time elapsed during editing, in seconds
    edit_time_elapsed = models.FloatField(default=0.0, blank=True)

    # Last modification user and datetime
    last_modified_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name='modified_labels', null=True, default=None)
    last_modified_datetime = models.DateTimeField(default=datetime.datetime.now)

    # Locked by user and expiry datetime
    locked_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name='locked_labels', null=True, default=None)
    lock_expiry_datetime = models.DateTimeField(default=datetime.datetime.now)

    # Manager
    objects = managers.LabelsManager()

    @property
    def labels_json(self):
        return json.loads(self.labels_json_str)

    @labels_json.setter
    def labels_json(self, label_js):
        self.labels_json_str = json.dumps(label_js)

    @property
    def is_empty(self):
        return self.labels_json_str == '[]'

    @property
    def label_classes(self):
        label_classes = [x['label_class']   for x in self.labels_json]
        return set(label_classes)

    def update_labels(self, labels_json, complete, time_elapsed, user, save=False, check_lock=False):
        # Verify time elapsed is within the bounds of possibility
        current_time = timezone.now()
        dt_since_last_mod = (current_time - self.last_modified_datetime).total_seconds()
        # Allow to either double the time since last modification or time since last modification plus 1 minute
        # to account for potential latency in delivery of last edit
        permitted_dt = max(dt_since_last_mod * 2.0, dt_since_last_mod + 60.0)
        permitted_time = self.edit_time_elapsed + permitted_dt
        if time_elapsed > permitted_time:
            print('WARNING: rejecting time_elapsed: '
                  'self.edit_time_elapsed={}, time_elapsed={}, permitted_time={}'.format(
                        self.edit_time_elapsed, time_elapsed, permitted_time
            ))
        elif time_elapsed >= self.edit_time_elapsed:
            self.edit_time_elapsed = time_elapsed

        if check_lock:
            if self.is_locked_to(user):
                raise LabelsLockedError
        self.labels_json = labels_json
        self.complete = complete
        if user.is_authenticated():
            self.last_modified_by = user
        else:
            self.last_modified_by = None
        self.last_modified_datetime = timezone.now()
        if save:
            self.save()

    def is_lock_active(self):
        return timezone.now() < self.lock_expiry_datetime and self.locked_by is not None

    def is_locked_to(self, user=None):
        lock_active = self.is_lock_active()
        if user is not None and not user.is_authenticated():
            user = None
        if user is None:
            return lock_active
        else:
            return lock_active and user != self.locked_by

    def lock(self, to_user, expire_after, save=False):
        if self.is_locked_to(to_user):
            raise ValueError('Cannot lock Labels(id={}) to user {}; is already locked'.format(
                self.id, to_user.username
            ))
        self.locked_by = to_user
        expiry = timezone.now() + expire_after
        self.lock_expiry_datetime = expiry
        if save:
            self.save()

    def refresh_lock(self, to_user, expire_after, save=False):
        if self.is_lock_active():
            if self.locked_by != to_user:
                raise ValueError('Cannot refresh lock Labels(id={}) for user {}; is already locked by {}'.format(
                    self.id, to_user.username, self.locked_by.username
                ))
        expiry = timezone.now() + expire_after
        self.lock_expiry_datetime = expiry
        if save:
            self.save()

    def unlock(self, from_user, save=False):
        if self.is_lock_active():
            if from_user != self.locked_by:
                raise ValueError('Cannot unlock Labels(id={}) from user {}, it is locked by {}'.format(
                    self.id, from_user.username, self.locked_by.username
                ))
            self.locked_by = None
            self.lock_expiry_datetime = timezone.now()
            if save:
                self.save()

    def __unicode__(self):
        if self.last_modified_by is not None:
            return 'Labels {} (last modified by {} at {})'.format(
                self.id, self.last_modified_by.username, self.last_modified_datetime)
        else:
            return 'Labels {}'.format(self.id)
