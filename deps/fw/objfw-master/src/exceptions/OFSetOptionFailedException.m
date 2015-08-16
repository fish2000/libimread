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

#import "OFSetOptionFailedException.h"
#import "OFString.h"
#import "OFStream.h"

@implementation OFSetOptionFailedException
+ (instancetype)exceptionWithStream: (OFStream*)stream
			      errNo: (int)errNo
{
	return [[[self alloc] initWithStream: stream
				       errNo: errNo] autorelease];
}

- init
{
	OF_INVALID_INIT_METHOD
}

- initWithStream: (OFStream*)stream
	   errNo: (int)errNo
{
	self = [super init];

	_stream = [stream retain];
	_errNo = errNo;

	return self;
}

- (void)dealloc
{
	[_stream release];

	[super dealloc];
}

- (OFString*)description
{
	return [OFString stringWithFormat:
	    @"Setting an option in a stream of type %@ failed: %@",
	    [_stream class], of_strerror(_errNo)];
}

- (OFStream*)stream
{
	OF_GETTER(_stream, true)
}

- (int)errNo
{
	return _errNo;
}
@end
