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

#import "OFInvalidJSONException.h"
#import "OFString.h"

@implementation OFInvalidJSONException
+ (instancetype)exceptionWithString: (OFString*)string
			       line: (size_t)line
{
	return [[[self alloc] initWithString: string
					line: line] autorelease];
}

- init
{
	OF_INVALID_INIT_METHOD
}

- initWithString: (OFString*)string
	    line: (size_t)line
{
	self = [super init];

	@try {
		_string = [string copy];
		_line = line;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (void)dealloc
{
	[_string release];

	[super dealloc];
}

- (OFString*)description
{
	return [OFString stringWithFormat:
	    @"The JSON representation is invalid in line %zu!", _line];
}

- (OFString*)string
{
	OF_GETTER(_string, true)
}

- (size_t)line
{
	return _line;
}
@end
