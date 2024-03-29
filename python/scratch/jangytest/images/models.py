from __future__ import unicode_literals

from django.db import models
from im.django import modelfields as imagemodels

from django.core.files.storage import default_storage

class Thingy(models.Model):
    
    class Meta:
        verbose_name = 'Image-Hosting Thingy'
        verbose_name_plural = 'Image-Hosting Thingees'
        abstract = False
    
    createdate = models.DateTimeField('Created on',
        auto_now_add=True,
        blank=True,
        editable=False)
    
    modifydate = models.DateTimeField('Last modified on',
        auto_now=True,
        blank=True,
        editable=False)
    
    w = models.IntegerField(verbose_name="width",
        editable=False,
        null=True)
    
    h = models.IntegerField(verbose_name="height",
        editable=False,
        null=True)
    
    image = imagemodels.ImageField(verbose_name="Image",
        storage=default_storage,
        null=True,
        blank=False,
        upload_to='uploads/',
        height_field='h',
        width_field='w',
        max_length=255)
    
    @property
    def image_struct(self):
        if hasattr(self, '_image_struct'):
            return self._image_struct
        if (not self.pk) or (not hasattr(self.image, 'path')):
            return None
        try:
            import im
            out = im.Image(self.image.path)
        except ImportError:
            return None
        except IOError:
            return None
        else:
            setattr(self, '_image_struct', out)
            return out
        return None
    
    @property
    def array_struct(self):
        if hasattr(self, '_array_struct'):
            return self._array_struct
        if (not self.pk) or (not hasattr(self.image, 'path')):
            return None
        try:
            import im
            out = im.Array(self.image.path)
        except ImportError:
            return None
        except IOError:
            return None
        else:
            setattr(self, '_array_struct', out)
            return out
        return None
    
    @property
    def suffix(self):
        if hasattr(self, '_suffix'):
            return self._suffix
        if (not self.pk) or (not hasattr(self.image, 'path')):
            return None
        try:
            import im
            out = im.detect(self.image.path)
        except ImportError:
            return None
        except IOError:
            return None
        else:
            setattr(self, '_suffix', out)
            return out
        return None
    
    @property
    def mimetype(self):
        if hasattr(self, '_mimetype'):
            return self._mimetype
        if (not self.pk) or (not hasattr(self.image, 'path')):
            return None
        try:
            import im
            out = im.mimetype(im.detect(self.image.path))
        except ImportError:
            return None
        except IOError:
            return None
        else:
            setattr(self, '_mimetype', out)
            return out
        return None