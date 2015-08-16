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

#define OF_CONSTANT_STRING_M

#include "config.h"

#include <stdlib.h>
#include <string.h>

#import "OFConstantString.h"
#import "OFString_UTF8.h"

#import "OFInitializationFailedException.h"
#import "OFInvalidEncodingException.h"
#import "OFOutOfMemoryException.h"

#if defined(OF_APPLE_RUNTIME) && !defined(__OBJC2__)
# import <objc/runtime.h>

struct {
	struct class *isa, *super_class;
	const char *name;
	long version, info, instance_size;
	struct ivar_list *ivars;
	struct method_list **method_lists;
	struct cache *cache;
	struct protocol_list *protocols;
	const char *ivar_layout;
	struct class_ext *ext;
} _OFConstantStringClassReference;
#endif

@interface OFString_const: OFString_UTF8
@end

@implementation OFString_const
+ alloc
{
	OF_UNRECOGNIZED_SELECTOR
}


- (void*)allocMemoryWithSize: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)allocMemoryWithSize: (size_t)size
		       count: (size_t)count
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)resizeMemory: (void*)pointer
		 size: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)resizeMemory: (void*)pointer
		 size: (size_t)size
		count: (size_t)count
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void)freeMemory: (void*)pointer
{
	OF_UNRECOGNIZED_SELECTOR
}

- retain
{
	return self;
}

- autorelease
{
	return self;
}

- (unsigned int)retainCount
{
	return OF_RETAIN_COUNT_MAX;
}

- (void)release
{
}

- (void)dealloc
{
	OF_UNRECOGNIZED_SELECTOR

	/* Get rid of a stupid warning */
	[super dealloc];
}
@end

@implementation OFConstantString
+ (void)load
{
#if defined(OF_APPLE_RUNTIME) && !defined(__OBJC2__)
	/*
	 * objc_setFutureClass suddenly stopped working as OFConstantString
	 * became more complex. So the only solution is to make
	 * _OFConstantStringClassRerence the actual class, but there is no
	 * objc_initializeClassPair in 10.5. However, objc_allocateClassPair
	 * does not register the new class with the subclass in the ObjC1
	 * runtime like the ObjC2 runtime does, so this workaround should be
	 * fine.
	 */
	Class class;

	if ((class = objc_allocateClassPair(self, "OFConstantString_hack",
	    0)) == NULL)
		@throw [OFInitializationFailedException
		    exceptionWithClass: self];
	memcpy(&_OFConstantStringClassReference, class,
	    sizeof(_OFConstantStringClassReference));
	free(class);
	objc_registerClassPair((Class)&_OFConstantStringClassReference);
#endif
}

- (void)finishInitialization
{
	@synchronized (self) {
		struct of_string_utf8_ivars *ivars;

		if (object_getClass(self) == [OFString_const class])
			return;

		if ((ivars = calloc(1, sizeof(*ivars))) == NULL)
			@throw [OFOutOfMemoryException
			    exceptionWithRequestedSize: sizeof(*ivars)];

		ivars->cString = _cString;
		ivars->cStringLength = _cStringLength;

		switch (of_string_utf8_check(ivars->cString,
		    ivars->cStringLength,
			&ivars->length)) {
			case 1:
				ivars->isUTF8 = true;
				break;
			case -1:
				free(ivars);
				@throw [OFInvalidEncodingException exception];
		}

		_cString = (char*)ivars;
		object_setClass(self, [OFString_const class]);
	}
}

+ alloc
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)allocMemoryWithSize: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)allocMemoryWithSize: (size_t)size
		       count: (size_t)count
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)resizeMemory: (void*)pointer
		 size: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)resizeMemory: (void*)pointer
		 size: (size_t)size
		count: (size_t)count
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void)freeMemory: (void*)pointer
{
	OF_UNRECOGNIZED_SELECTOR
}

- retain
{
	return self;
}

- autorelease
{
	return self;
}

- (unsigned int)retainCount
{
	return OF_RETAIN_COUNT_MAX;
}

- (void)release
{
}

- (void)dealloc
{
	OF_UNRECOGNIZED_SELECTOR

	/* Get rid of a stupid warning */
	[super dealloc];
}

/*
 * In all following methods, the constant string is converted to an
 * OFString_UTF8 and the message sent again.
 */

/* From protocol OFCopying */
- copy
{
	[self finishInitialization];

	return [self copy];
}

/* From protocol OFMutableCopying */
- mutableCopy
{
	[self finishInitialization];

	return [self mutableCopy];
}

/* From protocol OFComparing */
- (of_comparison_result_t)compare: (id <OFComparing>)object
{
	[self finishInitialization];

	return [self compare: object];
}

/* From OFObject, but reimplemented in OFString */
- (bool)isEqual: (id)object
{
	[self finishInitialization];

	return [self isEqual: object];
}

- (uint32_t)hash
{
	[self finishInitialization];

	return [self hash];
}

- (OFString*)description
{
	[self finishInitialization];

	return [self description];
}

/* From OFString */
- (const char*)UTF8String
{
	[self finishInitialization];

	return [self UTF8String];
}

- (size_t)getCString: (char*)cString_
	   maxLength: (size_t)maxLength
	    encoding: (of_string_encoding_t)encoding
{
	[self finishInitialization];

	return [self getCString: cString_
		      maxLength: maxLength
		       encoding: encoding];
}

- (const char*)cStringWithEncoding: (of_string_encoding_t)encoding
{
	[self finishInitialization];

	return [self cStringWithEncoding: encoding];
}

- (size_t)length
{
	[self finishInitialization];

	return [self length];
}

- (size_t)UTF8StringLength
{
	[self finishInitialization];

	return [self UTF8StringLength];
}

- (size_t)cStringLengthWithEncoding: (of_string_encoding_t)encoding
{
	[self finishInitialization];

	return [self cStringLengthWithEncoding: encoding];
}

- (of_comparison_result_t)caseInsensitiveCompare: (OFString*)otherString
{
	[self finishInitialization];

	return [self caseInsensitiveCompare: otherString];
}

- (of_unichar_t)characterAtIndex: (size_t)index
{
	[self finishInitialization];

	return [self characterAtIndex: index];
}

- (void)getCharacters: (of_unichar_t*)buffer
	      inRange: (of_range_t)range
{
	[self finishInitialization];

	[self getCharacters: buffer
		    inRange: range];
}

- (of_range_t)rangeOfString: (OFString*)string
{
	[self finishInitialization];

	return [self rangeOfString: string];
}

- (of_range_t)rangeOfString: (OFString*)string
		    options: (int)options
{
	[self finishInitialization];

	return [self rangeOfString: string
			   options: options];
}

- (of_range_t)rangeOfString: (OFString*)string
		    options: (int)options
		      range: (of_range_t)range
{
	[self finishInitialization];

	return [self rangeOfString: string
			   options: options
			     range: range];
}

- (bool)containsString: (OFString*)string
{
	[self finishInitialization];

	return [self containsString: string];
}

- (OFString*)substringWithRange: (of_range_t)range
{
	[self finishInitialization];

	return [self substringWithRange: range];
}

- (OFString*)stringByAppendingString: (OFString*)string
{
	[self finishInitialization];

	return [self stringByAppendingString: string];
}

- (OFString*)stringByAppendingFormat: (OFConstantString*)format
			   arguments: (va_list)arguments
{
	[self finishInitialization];

	return [self stringByAppendingFormat: format
				   arguments: arguments];
}

- (OFString*)stringByAppendingPathComponent: (OFString*)component
{
	[self finishInitialization];

	return [self stringByAppendingPathComponent: component];
}

- (OFString*)stringByPrependingString: (OFString*)string
{
	[self finishInitialization];

	return [self stringByPrependingString: string];
}

- (OFString*)stringByReplacingOccurrencesOfString: (OFString*)string
				       withString: (OFString*)replacement
{
	[self finishInitialization];

	return [self stringByReplacingOccurrencesOfString: string
					       withString: replacement];
}

- (OFString*)stringByReplacingOccurrencesOfString: (OFString*)string
				       withString: (OFString*)replacement
					  options: (int)options
					    range: (of_range_t)range
{
	[self finishInitialization];

	return [self stringByReplacingOccurrencesOfString: string
					       withString: replacement
						  options: options
						    range: range];
}

- (OFString*)uppercaseString
{
	[self finishInitialization];

	return [self uppercaseString];
}

- (OFString*)lowercaseString
{
	[self finishInitialization];

	return [self lowercaseString];
}

- (OFString*)capitalizedString
{
	[self finishInitialization];

	return [self capitalizedString];
}

- (OFString*)stringByDeletingLeadingWhitespaces
{
	[self finishInitialization];

	return [self stringByDeletingLeadingWhitespaces];
}

- (OFString*)stringByDeletingTrailingWhitespaces
{
	[self finishInitialization];

	return [self stringByDeletingTrailingWhitespaces];
}

- (OFString*)stringByDeletingEnclosingWhitespaces
{
	[self finishInitialization];

	return [self stringByDeletingEnclosingWhitespaces];
}

- (bool)hasPrefix: (OFString*)prefix
{
	[self finishInitialization];

	return [self hasPrefix: prefix];
}

- (bool)hasSuffix: (OFString*)suffix
{
	[self finishInitialization];

	return [self hasSuffix: suffix];
}

- (OFArray*)componentsSeparatedByString: (OFString*)delimiter
{
	[self finishInitialization];

	return [self componentsSeparatedByString: delimiter];
}

- (OFArray*)componentsSeparatedByString: (OFString*)delimiter
				options: (int)options
{
	[self finishInitialization];

	return [self componentsSeparatedByString: delimiter
					 options: options];
}

- (OFArray*)pathComponents
{
	[self finishInitialization];

	return [self pathComponents];
}

- (OFString*)lastPathComponent
{
	[self finishInitialization];

	return [self lastPathComponent];
}

- (OFString*)stringByDeletingLastPathComponent
{
	[self finishInitialization];

	return [self stringByDeletingLastPathComponent];
}

- (intmax_t)decimalValue
{
	[self finishInitialization];

	return [self decimalValue];
}

- (uintmax_t)hexadecimalValue
{
	[self finishInitialization];

	return [self hexadecimalValue];
}

- (float)floatValue
{
	[self finishInitialization];

	return [self floatValue];
}

- (double)doubleValue
{
	[self finishInitialization];

	return [self doubleValue];
}

- (const of_unichar_t*)characters
{
	[self finishInitialization];

	return [self characters];
}

- (const of_char16_t*)UTF16String
{
	[self finishInitialization];

	return [self UTF16String];
}

- (const of_char16_t*)UTF16StringWithByteOrder: (of_byte_order_t)byteOrder
{
	[self finishInitialization];

	return [self UTF16StringWithByteOrder: byteOrder];
}

- (size_t)UTF16StringLength
{
	[self finishInitialization];

	return [self UTF16StringLength];
}

- (const of_char32_t*)UTF32String
{
	[self finishInitialization];

	return [self UTF32String];
}

- (const of_char32_t*)UTF32StringWithByteOrder: (of_byte_order_t)byteOrder
{
	[self finishInitialization];

	return [self UTF32StringWithByteOrder: byteOrder];
}

- (void)writeToFile: (OFString*)path
{
	[self finishInitialization];

	[self writeToFile: path];
}

- (void)writeToFile: (OFString*)path
	   encoding: (of_string_encoding_t)encoding
{
	[self finishInitialization];

	[self writeToFile: path
		 encoding: encoding];
}

#ifdef OF_HAVE_BLOCKS
- (void)enumerateLinesUsingBlock: (of_string_line_enumeration_block_t)block
{
	[self finishInitialization];

	[self enumerateLinesUsingBlock: block];
}
#endif
@end
