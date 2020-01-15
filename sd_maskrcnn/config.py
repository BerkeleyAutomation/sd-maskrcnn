"""
Copyright ©2019. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Author: Mike Danielczuk
"""

from mrcnn.config import Config

class MaskConfig(Config):
  """Configuration for training SD Mask RCNN.
  Derives from the base Config class and overrides values specific
  to SD Mask RCNN.
  """
  NAME = 'my_sdmaskrcnn'

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # background + object
  USE_MINI_MASK = False

  # Use smaller anchors because our image and objects are small
  # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

  def __init__(self, config):
    # Overriding things here.
    for x in config:
      setattr(self, x.upper(), config[x])
    super(MaskConfig, self).__init__()
