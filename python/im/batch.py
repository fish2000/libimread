
from im.im import (
    hybridimage_check,
    image_check,
    buffer_check,
    imagebuffer_check,
    array_check,
    arraybuffer_check,
    batch_check,
    batchiterator_check,
    HybridImage,
    Image, Array,
    Buffer, Batch)

types = (Image, Array, Buffer, Batch)

class constrain(type):
    
    def __new__(cls, name, bases, attrs):
        # super_new bit cribbed from Django:
        super_new = super(constrain, cls).__new__
        newcls = super_new(cls, name, bases, attrs)
        # if newcls.json_name != 'field':
        #     CONSTRAINT_DMV.append(newcls)
        return newcls
    
    def to_type(cls, RequiredType):
        if RequiredType not in types:
            raise TypeError("Unknown required_type")
        def checker(self, obj):
            return RequiredType.check(obj)
        return checker


class ConstraintBase(object):
    json_name = u'constraint'
    
    @property
    def name(self):
        return str(self.json_name)
    
    # def check(self, obj):
    #     return True


class Constraint(ConstraintBase):
    __metaclass__ = constrain
    
    def disqualify(self, obj_list):
        pass # specifically DO NOT RAISE

class TypeConstraint(Constraint):
    json_name = u'type'
    
    class __metaclass__(constrain):
        pass

