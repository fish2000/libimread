
import numpy
from PIL.Image import Image as PILImage

from im.im import (
    # hybridimage_check,
    # image_check,
    # buffer_check,
    # imagebuffer_check,
    # array_check,
    # arraybuffer_check,
    # batch_check,
    # batchiterator_check,
    HybridImage,
    Image, Array,
    Buffer)

from im.im import Batch as NativeBatch

types = (HybridImage,
         Image, Array, Buffer,
         PILImage,
         numpy.ndarray)


class ConstraintBase(object):
    
    json_name = u'constraint'
    valid_modifiers = [None]
    
    @property
    def name(self):
        return str(self.json_name)
    
    @property
    def modifiers(self):
        return tuple(self.valid_modifiers)

class TypeConstraint(ConstraintBase):
    json_name = u'item_type'
    valid_modifiers = ['not', None]

class TypeConstraintList(ConstraintBase):
    json_name = u'item_types'
    valid_modifiers = ['not', None]

class BatchSizeConstraint(ConstraintBase):
    json_name = u'batch_size'
    valid_modifiers = ['lt', 'gt', 'lte', 'gte', None]

class WidthConstraint(ConstraintBase):
    json_name = u'width'
    valid_modifiers = ['lt', 'gt', 'lte', 'gte', None]

class HeightConstraint(ConstraintBase):
    json_name = u'height'
    valid_modifiers = ['lt', 'gt', 'lte', 'gte', None]

class PlanesConstraint(ConstraintBase):
    json_name = u'planes'
    valid_modifiers = ['lt', 'gt', 'lte', 'gte', None]

class SizeConstraint(ConstraintBase):
    json_name = u'size'
    valid_modifiers = ['lt', 'gt', 'lte', 'gte', None]

class DimensionsConstraint(ConstraintBase):
    json_name = u'dimensions'
    valid_modifiers = ['lt', 'gt', 'lte', 'gte', None]

class constrain(object):
    """ Declarative constraint operator. """
    
    class StipulationError(Exception):
        """ Raised when a nonsensical constraint stipulation is parsed.
        """
        pass
    
    NOTIONS = ('item_type', 'item_types',
               'batch_size',
               'width', 'height', 'planes', 'size', 'dimensions',
               'input_format', 'input_formats',
               'output_format', 'output_formats',
               'unique_items',
               'value',
               'alpha')
    
    MODIFIERS = ('not', 'lt', 'gt', 'lte', 'gte', None)
    DIVIDER = '__'
    
    @staticmethod
    def reconstruct(key, value):
        return "%s=%s" % (key, value)
    
    def __init__(self, *args, **stipulations):
        
        if len(stipulations) < 1:
            raise self.StipulationError("no stipulations in constraint")
        
        self.internal = {}
        
        for stipulation, stipulation_value in stipulations.items():
            segments = stipulation.count(self.DIVIDER)
            if (segments == 0):
                notion = stipulation
                custom_value = modifier = None
            elif (segments == 1):
                notion, modifier = stipulation.split(self.DIVIDER)
                custom_value = None
            elif (segments == 2):
                notion, custom_value, modifier = stipulation.split(self.DIVIDER)
            else:
                raise self.StipulationError("too many segments: '%s'" %
                                            self.reconstruct(stipulation,
                                                             stipulation_value))
            if notion not in self.NOTIONS:
                raise self.StipulationError("unknown constraint notion: '%s'" %
                                            self.reconstruct(stipulation,
                                                             stipulation_value))
            if modifier not in self.MODIFIERS:
                raise self.StipulationError("unknown constraint modifier: '%s'" %
                                            self.reconstruct(stipulation,
                                                             stipulation_value))
            if stipulation_value is None:
                raise self.StipulationError("value is None: '%s'" %
                                            self.reconstruct(stipulation,
                                                             stipulation_value))
            
            self.internal[notion] = dict(value=stipulation_value,
                                         custom_value=custom_value,
                                         modifier=modifier)
    
    def get(self, notion):
        return self.internal.get(notion)
    
    def notions(self):
        return self.internal.keys()
    
    def all(self):
        return self.internal


class Constrainer(type):
    pass

class ConstrainedBatch(NativeBatch, object):
    __metaclass__ = Constrainer
    
    class ConstraintError(Exception):
        """ Raised when an attempt is made to add an image to a
            ConstrainedBatch whose constraints forbid such an image
            to be added to such a batch.
        """
        pass

class BatchItemProxy(object):
    pass

class FirstItem(BatchItemProxy):
    pass

class LastItem(BatchItemProxy):
    pass

class AllItems(BatchItemProxy):
    pass

class Batch(ConstrainedBatch):
    
    constrain(item_type=Image)                      # accept ONLY Image item types
    constrain(item_types=(Image, Array))            # accepts Image OR Array types
    constrain(item_types__not=(Image, Array))       # accepts all types EXCEPT Image OR Array types
    constrain(width__gte=1024, height__gte=768)     # width >= 1024 AND height >= 768
    constrain(batch_size__lte=10)
    constrain(planes=AllItems.planes)
    constrain(size=FirstItem.size)                  # size = (width, height)
    constrain(dimensions=FirstItem.dimensions)      # dimensions = (width, height, planes)
    constrain(input_format='gif')
    constrain(input_formats=('gif', 'tif'))         # gif OR tif
    constrain(output_format='gif')
    constrain(output_formats=('gif', 'tif'))        # gif OR tif
    constrain(input_formats__not=('gif', 'tif'))    # All formats EXCEPT gif OR tif
    constrain(unique_items=True)                    # Don't accept duplicate items
    constrain(unique_items=False)                   # Allow duplicate items
    constrain(value__entropy__gte=4.5)              # Custom value stipulation (entropy >= 4.5)
    constrain(alpha=True)                           # Require alpha channels (False forbids images with alpha channels)
    
        
    


# class constrain(type):
#
#     def __new__(cls, name, bases, attrs):
#         # super_new bit cribbed from Django:
#         super_new = super(constrain, cls).__new__
#         newcls = super_new(cls, name, bases, attrs)
#         # if newcls.json_name != 'field':
#         #     CONSTRAINT_DMV.append(newcls)
#         return newcls
#
#     def to_type(cls, RequiredType):
#         if RequiredType not in types:
#             raise TypeError("Unknown required_type")
#         def checker(self, obj):
#             return RequiredType.check(obj)
#         return checker


# class ConstraintBase(object):
#     json_name = u'constraint'
#
#     @property
#     def name(self):
#         return str(self.json_name)
#
#     # def check(self, obj):
#     #     return True


# class Constraint(ConstraintBase):
#     __metaclass__ = constrain
#
#     def disqualify(self, obj_list):
#         pass # specifically DO NOT RAISE
#
# class TypeConstraint(Constraint):
#     json_name = u'type'
#
#     class __metaclass__(constrain):
#         pass
#
