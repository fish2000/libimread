/*
 * Copyright (c) 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015
 *   Jonathan Schleifer <js@webkeks.org>
 *
 * All rights reserved.
 *
 * This file is part of ObjFW. It may be distributed under the terms of the
 * Q Public License 1.0, which can be found in the file LICENSE.QPL included in
 * the packaging of this file.
 *
 * Alternatively, it may be distributed under the terms of the GNU General
 * Public License, either version 2 or 3, which can be found in the file
 * LICENSE.GPLv2 or LICENSE.GPLv3 respectively included in the packaging of this
 * file.
 */

#include "config.h"

#include <string.h>
#include <ctype.h>

#import "OFIntrospection.h"
#import "OFString.h"
#import "OFArray.h"

#import "OFInitializationFailedException.h"

@implementation OFMethod
#if defined(OF_OBJFW_RUNTIME)
- (instancetype)OF_initWithMethod: (struct objc_method*)method
{
	self = [super init];

	@try {
		_selector = (SEL)&method->sel;
		_name = [[OFString alloc]
		    initWithUTF8String: sel_getName(_selector)];
		_typeEncoding = method->sel.types;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#elif defined(OF_APPLE_RUNTIME)
- (instancetype)OF_initWithMethod: (Method)method
{
	self = [super init];

	@try {
		_selector = method_getName(method);
		_name = [[OFString alloc]
		    initWithUTF8String: sel_getName(_selector)];
		_typeEncoding = method_getTypeEncoding(method);
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#else
# error Invalid ObjC runtime!
#endif

- init
{
	OF_INVALID_INIT_METHOD
}

- (void)dealloc
{
	[_name release];

	[super dealloc];
}

- (SEL)selector
{
	return _selector;
}

- (OFString*)name
{
	OF_GETTER(_name, true)
}

- (const char*)typeEncoding
{
	return _typeEncoding;
}

- (OFString*)description
{
	return [OFString stringWithFormat: @"<%@: %@ [%s]>",
					   [self class], _name, _typeEncoding];
}

- (bool)isEqual: (id)object
{
	OFMethod *method;

	if (![object isKindOfClass: [OFMethod class]])
		return false;

	method = object;

	if (!sel_isEqual(method->_selector, _selector))
		return false;

	if (![method->_name isEqual: _name])
		return false;

	if ((method->_typeEncoding == NULL && _typeEncoding != NULL) ||
	    (method->_typeEncoding != NULL && _typeEncoding == NULL))
		return false;

	if (method->_typeEncoding != NULL && _typeEncoding != NULL &&
	    strcmp(method->_typeEncoding, _typeEncoding) != 0)
		return false;

	return true;
}

- (uint32_t)hash
{
	uint32_t hash;

	OF_HASH_INIT(hash);

	OF_HASH_ADD_HASH(hash, [_name hash]);

	if (_typeEncoding != NULL) {
		size_t i, length;

		length = strlen(_typeEncoding);
		for (i = 0; i < length; i++)
			OF_HASH_ADD(hash, _typeEncoding[i]);
	}

	OF_HASH_FINALIZE(hash);

	return hash;
}
@end

@implementation OFProperty
#if defined(OF_OBJFW_RUNTIME)
- (instancetype)OF_initWithProperty: (struct objc_property*)property
{
	self = [super init];

	@try {
		_name = [[OFString alloc] initWithUTF8String: property->name];
		_attributes =
		    property->attributes | (property->extended_attributes << 8);

		if (property->getter.name != NULL)
			_getter = [[OFString alloc]
			    initWithUTF8String: property->getter.name];
		if (property->setter.name != NULL)
			_setter = [[OFString alloc]
			    initWithUTF8String: property->setter.name];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#elif defined(OF_APPLE_RUNTIME)
- (instancetype)OF_initWithProperty: (objc_property_t)property
{
	self = [super init];

	@try {
		const char *attributes;

		_name = [[OFString alloc]
		    initWithUTF8String: property_getName(property)];

		if ((attributes = property_getAttributes(property)) == NULL)
			@throw [OFInitializationFailedException
			    exceptionWithClass: [self class]];

		while (*attributes != '\0') {
			const char *start;

			switch (*attributes) {
			case 'R':
				_attributes |= OF_PROPERTY_READONLY;
				attributes++;
				break;
			case 'C':
				_attributes |= OF_PROPERTY_COPY;
				attributes++;
				break;
			case '&':
				_attributes |= OF_PROPERTY_RETAIN;
				attributes++;
				break;
			case 'N':
				_attributes |= OF_PROPERTY_NONATOMIC;
				attributes++;
				break;
			case 'G':
				start = ++attributes;

				if (_getter != nil)
					@throw [OFInitializationFailedException
					    exceptionWithClass: [self class]];

				while (*attributes != ',' &&
				    *attributes != '\0')
					attributes++;

				_getter = [[OFString alloc]
				    initWithUTF8String: start
						length: attributes - start];

				break;
			case 'S':
				start = ++attributes;

				if (_setter != nil)
					@throw [OFInitializationFailedException
					    exceptionWithClass: [self class]];

				while (*attributes != ',' &&
				    *attributes != '\0')
					attributes++;

				_setter = [[OFString alloc]
				    initWithUTF8String: start
						length: attributes - start];

				break;
			case 'D':
				_attributes |= OF_PROPERTY_DYNAMIC;
				attributes++;
				break;
			case 'W':
				_attributes |= OF_PROPERTY_WEAK;
				attributes++;
				break;
			case 'P':
				attributes++;
				break;
			case 'T':
			case 't':
				while (*attributes != ',' &&
				    *attributes != '\0')
					attributes++;
				break;
			default:
				@throw [OFInitializationFailedException
				    exceptionWithClass: [self class]];
			}

			if (*attributes != ',' && *attributes != '\0')
				@throw [OFInitializationFailedException
				    exceptionWithClass: [self class]];

			if (*attributes != '\0')
				attributes++;
		}

		if (!(_attributes & OF_PROPERTY_READONLY))
			_attributes |= OF_PROPERTY_READWRITE;

		if (!(_attributes & OF_PROPERTY_COPY) &&
		    !(_attributes & OF_PROPERTY_RETAIN))
			_attributes |= OF_PROPERTY_ASSIGN;

		if (!(_attributes & OF_PROPERTY_NONATOMIC))
			_attributes |= OF_PROPERTY_ATOMIC;

		if (!(_attributes & OF_PROPERTY_DYNAMIC))
			_attributes |= OF_PROPERTY_SYNTHESIZED;

		if (_getter == nil)
			_getter = [_name copy];

		if ((_attributes & OF_PROPERTY_READWRITE) && _setter == nil) {
			of_unichar_t first = [_name characterAtIndex: 0];
			OFMutableString *tmp = [_name mutableCopy];
			_setter = tmp;

			if (first < 0x80) {
				[tmp setCharacter: toupper((int)first)
					  atIndex: 0];
			}

			[tmp prependString: @"set"];

			[tmp makeImmutable];
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#else
# error Invalid ObjC runtime!
#endif

- (void)dealloc
{
	[_name release];
	[_getter release];
	[_setter release];

	[super dealloc];
}

- (OFString*)name
{
	OF_GETTER(_name, true)
}

- (unsigned)attributes
{
	return _attributes;
}

- (OFString*)getter
{
	OF_GETTER(_getter, true)
}

- (OFString*)setter
{
	OF_GETTER(_setter, true)
}

- (OFString*)description
{
	return [OFString
	    stringWithFormat: @"<%@: %@\n"
			      @"\tAttributes = 0x%03X\n"
			      @"\tGetter = %@\n"
			      @"\tSetter = %@\n"
			      @">",
			      [self class], _name, _attributes,
			      _getter, _setter];
}

- (bool)isEqual: (id)object
{
	OFProperty *otherProperty;

	if ([object isKindOfClass: [OFProperty class]])
		return false;

	otherProperty = object;

	if (![otherProperty->_name isEqual: _name])
		return false;
	if (otherProperty->_attributes != _attributes)
		return false;
	if (![otherProperty->_getter isEqual: _getter])
		return false;
	if (![otherProperty->_setter isEqual: _setter])
		return false;

	return true;
}

- (uint32_t)hash
{
	uint32_t hash;

	OF_HASH_INIT(hash);

	OF_HASH_ADD_HASH(hash, [_name hash]);
	OF_HASH_ADD(hash, (_attributes & 0xFF00) >> 8);
	OF_HASH_ADD(hash, _attributes & 0xFF);
	OF_HASH_ADD_HASH(hash, [_getter hash]);
	OF_HASH_ADD_HASH(hash, [_setter hash]);

	OF_HASH_FINALIZE(hash);

	return hash;
}
@end

@implementation OFInstanceVariable
#if defined(OF_OBJFW_RUNTIME)
- (instancetype)OF_initWithIvar: (struct objc_ivar*)ivar
{
	self = [super init];

	@try {
		_name = [[OFString alloc] initWithUTF8String: ivar->name];
		_typeEncoding = ivar->type;
		_offset = ivar->offset;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#elif defined(OF_APPLE_RUNTIME)
- (instancetype)OF_initWithIvar: (Ivar)ivar
{
	self = [super init];

	@try {
		_name = [[OFString alloc]
		    initWithUTF8String: ivar_getName(ivar)];
		_typeEncoding = ivar_getTypeEncoding(ivar);
		_offset = ivar_getOffset(ivar);
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#else
# error Invalid ObjC runtime!
#endif

- init
{
	OF_INVALID_INIT_METHOD
}

- (void)dealloc
{
	[_name release];

	[super dealloc];
}

- (OFString*)name
{
	OF_GETTER(_name, true);
}

- (ptrdiff_t)offset
{
	return _offset;
}

- (const char*)typeEncoding
{
	return _typeEncoding;
}

- (OFString*)description
{
	return [OFString stringWithFormat:
	    @"<OFInstanceVariable: %@ [%s] @ 0x%tx>",
	    _name, _typeEncoding, _offset];
}
@end

@implementation OFIntrospection
+ (instancetype)introspectionWithClass: (Class)class
{
	return [[[self alloc] initWithClass: class] autorelease];
}

- initWithClass: (Class)class
{
	self = [super init];

	@try {
#if defined(OF_OBJFW_RUNTIME)
		struct objc_method_list *methodList;
		struct objc_property_list *propertyList;
#elif defined(OF_APPLE_RUNTIME)
		Method *methodList;
		objc_property_t *propertyList;
		Ivar *ivarList;
		unsigned count;
#endif
		unsigned i;
		void *pool;

		_classMethods = [[OFMutableArray alloc] init];
		_instanceMethods = [[OFMutableArray alloc] init];
		_properties = [[OFMutableArray alloc] init];
		_instanceVariables = [[OFMutableArray alloc] init];

#if defined(OF_OBJFW_RUNTIME)
		for (methodList = object_getClass(class)->methodlist;
		    methodList != NULL; methodList = methodList->next) {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < methodList->count; i++)
				[_classMethods addObject: [[[OFMethod alloc]
				    OF_initWithMethod:
				    &methodList->methods[i]] autorelease]];

			objc_autoreleasePoolPop(pool);
		}

		for (methodList = class->methodlist; methodList != NULL;
		    methodList = methodList->next) {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < methodList->count; i++)
				[_instanceMethods addObject: [[[OFMethod alloc]
				    OF_initWithMethod:
				    &methodList->methods[i]] autorelease]];

			objc_autoreleasePoolPop(pool);
		}

		for (propertyList = class->properties; propertyList != NULL;
		    propertyList = propertyList->next) {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < propertyList->count; i++)
				[_properties addObject: [[[OFProperty alloc]
				    OF_initWithProperty:
				    &propertyList->properties[i]] autorelease]];

			objc_autoreleasePoolPop(pool);
		}

		if (class->ivars != NULL) {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < class->ivars->count; i++)
				[_instanceVariables addObject:
				    [[[OFInstanceVariable alloc]
				    OF_initWithIvar:
				    &class->ivars->ivars[i]] autorelease]];

			objc_autoreleasePoolPop(pool);
		}
#elif defined(OF_APPLE_RUNTIME)
		methodList = class_copyMethodList(object_getClass(class),
		    &count);
		@try {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < count; i++)
				[_classMethods addObject: [[[OFMethod alloc]
				    OF_initWithMethod: methodList[i]]
				    autorelease]];

			objc_autoreleasePoolPop(pool);
		} @finally {
			free(methodList);
		}

		methodList = class_copyMethodList(class, &count);
		@try {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < count; i++)
				[_instanceMethods addObject: [[[OFMethod alloc]
				    OF_initWithMethod: methodList[i]]
				    autorelease]];

			objc_autoreleasePoolPop(pool);
		} @finally {
			free(methodList);
		}

		propertyList = class_copyPropertyList(class, &count);
		@try {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < count; i++)
				[_properties addObject: [[[OFProperty alloc]
				    OF_initWithProperty: propertyList[i]]
				    autorelease]];

			objc_autoreleasePoolPop(pool);
		} @finally {
			free(propertyList);
		}

		ivarList = class_copyIvarList(class, &count);
		@try {
			pool = objc_autoreleasePoolPush();

			for (i = 0; i < count; i++)
				[_instanceVariables addObject:
				    [[[OFInstanceVariable alloc]
				    OF_initWithIvar: ivarList[i]] autorelease]];

			objc_autoreleasePoolPop(pool);
		} @finally {
			free(ivarList);
		}
#else
# error Invalid ObjC runtime!
#endif

		[_classMethods makeImmutable];
		[_instanceMethods makeImmutable];
		[_properties makeImmutable];
		[_instanceVariables makeImmutable];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- init
{
	OF_INVALID_INIT_METHOD
}

- (void)dealloc
{
	[_classMethods release];
	[_instanceMethods release];
	[_instanceVariables release];

	[super dealloc];
}

- (OFArray*)classMethods
{
	OF_GETTER(_classMethods, true)
}

- (OFArray*)instanceMethods
{
	OF_GETTER(_instanceMethods, true)
}

- (OFArray*)properties
{
	OF_GETTER(_properties, true)
}

- (OFArray*)instanceVariables
{
	OF_GETTER(_instanceVariables, true)
}
@end
