import os, mimetypes, json, datetime
from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from image_labelling_tool import labelling_tool
from image_labelling_tool import models as lt_models
from ... import models

class Command(BaseCommand):
    help = 'Populates the image database from a directory'

    def add_arguments(self, parser):
        parser.add_argument('dir', type=str)

    def handle(self, *args, **options):
        images_dir = options['dir']
        image_and_label_files = []
        for filename in os.listdir(images_dir):
            path = os.path.join(images_dir, filename)
            if os.path.isfile(path):
                mt, encoding = mimetypes.guess_type(path)
                if mt is not None and mt.startswith('image/'):
                    image_path = path
                    labels_path = os.path.splitext(path)[0] + '__labels.json'
                    if os.path.exists(labels_path) and os.path.isfile(labels_path):
                        image_and_label_files.append((image_path, labels_path))
                    else:
                        image_and_label_files.append((image_path, None))

        for image_path, labels_path in image_and_label_files:
            if labels_path is not None:
                self.stdout.write('Adding image {} with labels from {}'.format(image_path, labels_path))
                wrapped_labels = json.load(open(labels_path, 'r'))
                labels, complete = labelling_tool.PersistentLabelledImage._unwrap_labels(
                    image_path, wrapped_labels)
                complete = complete if isinstance(complete, bool) else False
                labels_model = lt_models.Labels(
                    labels_json_str=json.dumps(labels.labels_json), complete=complete,
                    creation_date=datetime.date.today())
                labels_model.save()
            else:
                self.stdout.write('Adding image {}'.format(image_path))
                labels_model = lt_models.Labels(creation_date=datetime.date.today())
                labels_model.save()

            image_model = models.ImageWithLabels(labels=labels_model)
            image_model.image.save(os.path.basename(image_path),
                                   File(open(image_path, 'rb')))
            image_model.save()
