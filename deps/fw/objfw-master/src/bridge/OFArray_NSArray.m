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

#import <Foundation/NSArray.h>

#import "OFArray_NSArray.h"
#import "NSBridging.h"

#import "OFInitializationFailedException.h"
#import "OFOutOfRangeException.h"

@implementation OFArray_NSArray
- initWithNSArray: (NSArray*)array
{
	self = [super init];

	@try {
		if (array == nil)
			@throw [OFInitializationFailedException
			    exceptionWithClass: [self class]];

		_array = [array retain];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (id)objectAtIndex: (size_t)index
{
	id object;

	if (index > NSUIntegerMax)
		@throw [OFOutOfRangeException exception];

	object = [_array objectAtIndex: index];

	if ([(NSObject*)object conformsToProtocol: @protocol(NSBridging)])
		return [object OFObject];

	return object;
}

- (size_t)count
{
	return [_array count];
}
@end
