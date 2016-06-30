
from __future__ import unicode_literals

import sys
from django.forms.fields import FileField

from django.core.exceptions import ValidationError
from django.utils import six
from django.utils.translation import ugettext_lazy as _

class ImageField(FileField):
    default_error_messages = {
        'invalid_image': _(
            "Upload a valid image. The file you uploaded was either not an "
            "image or a corrupted image."
        ),
    }

    def to_python(self, data):
        """
        Checks that the file-upload field data contains a valid image (GIF, JPG,
        PNG, possibly others -- whatever libimread supports).
        """
        f = super(ImageField, self).to_python(data)
        if f is None:
            return None
        
        import im
        
        # We need to get a file object for libimread.
        # We might have a path or we might have to read the data into memory.
        is_blob = True
        if hasattr(data, 'temporary_file_path'):
            imagefile = data.temporary_file_path()
            is_blob = False
        else:
            if hasattr(data, 'read'):
                imagefile = data.read()
            else:
                imagefile = data['content']
        
        try:
            image = im.Image(imagefile, is_blob=is_blob)
            f.image = image
            f.content_type = im.mimetype(im.detect(imagefile, is_blob=is_blob))
        except Exception:
            # libimread doesn't recognize it as an image.
            six.reraise(ValidationError, ValidationError(
                self.error_messages['invalid_image'],
                code='invalid_image',
            ), sys.exc_info()[2])
        
        if hasattr(f, 'seek') and callable(f.seek):
            f.seek(0)
        return f
