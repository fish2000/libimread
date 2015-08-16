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

#import "OFNotImplementedException.h"
#import "OFString.h"

@implementation OFNotImplementedException
+ (instancetype)exceptionWithSelector: (SEL)selector
			       object: (id)object
{
	return [[[self alloc] initWithSelector: selector
					object: object] autorelease];
}

- init
{
	OF_INVALID_INIT_METHOD
}

- initWithSelector: (SEL)selector
	    object: (id)object
{
	self = [super init];

	_selector = selector;
	_object = [object retain];

	return self;
}

- (void)dealloc
{
	[_object release];

	[super dealloc];
}

- (OFString*)description
{
	return [OFString stringWithFormat:
	    @"The selector %s is not understood by an object of type %@ or not "
	    @"(fully) implemented!", sel_getName(_selector), [_object class]];
}

- (SEL)selector
{
	return _selector;
}

- (id)object
{
	OF_GETTER(_object, true)
}
@end
