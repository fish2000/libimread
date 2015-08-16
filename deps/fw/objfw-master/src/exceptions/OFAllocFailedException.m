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

#import "OFAllocFailedException.h"
#import "OFString.h"

@implementation OFAllocFailedException
+ alloc
{
	OF_UNRECOGNIZED_SELECTOR
}

- init
{
	OF_INVALID_INIT_METHOD
}

- (void*)allocMemoryWithSize: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)allocMemoryForNItems: (size_t)nitems
                     withSize: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)resizeMemory: (void*)ptr
	       toSize: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void*)resizeMemory: (void*)ptr
	     toNItems: (size_t)nitems
	     withSize: (size_t)size
{
	OF_UNRECOGNIZED_SELECTOR
}

- (void)freeMemory: (void*)ptr
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

- (OFString*)description
{
	return @"Allocating an object failed!";
}
@end
