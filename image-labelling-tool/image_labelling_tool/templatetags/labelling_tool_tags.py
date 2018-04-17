import uuid, json

from django import template
from django.utils.html import format_html

from image_labelling_tool import labelling_tool as lt

register = template.Library()


def _update_config(dest, src):
    if isinstance(src, dict):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dest.get(key), dict):
                print('Updating {}...'.format(key))
                _update_config(dest[key], src[key])
            else:
                dest[key] = value


@register.simple_tag
def labelling_tool_scripts():
    script_urls = lt.js_file_urls('/static/labelling_tool/')
    script_tags = ''.join(['<script src="{}"></script>\n'.format(url) for url in script_urls])
    return format_html(script_tags)

@register.inclusion_tag('inline/labelling_tool.html')
def labelling_tool(width, height, label_classes, image_descriptors, initial_image_index,
                   labelling_tool_url, enable_locking, config=None):
    tool_id = uuid.uuid4()
    if config is None:
        config = {}
    return {'tool_id': str(tool_id),
            'width': width,
            'height': height,
            'label_classes': json.dumps(label_classes),
            'image_descriptors': json.dumps(image_descriptors),
            'initial_image_index': str(initial_image_index),
            'labelling_tool_url': labelling_tool_url,
            'enable_locking': enable_locking,
            'config': json.dumps(config),
            }

@register.inclusion_tag('inline/instructions.html')
def labelling_tool_instructions(config=None):
    print('labelling_tool_instructions: {}'.format(config))
    tools = {
        'imageSelector': True,
        'labelClassSelector': True,
        'brushSelect': True,
        'drawPointLabel': True,
        'drawBoxLabel': True,
        'drawPolyLabel': True,
        'compositeLabel': True,
        'groupLabel': True,
        'deleteLabel': True,
        'deleteConfig': {
            'typePermissions': {
                'point': True,
                'box': True,
                'polygon': True,
                'composite': True,
                'group': True,
            }
        }
    }
    if config is not None:
        _update_config(tools, config.get('tools'))

    return tools
