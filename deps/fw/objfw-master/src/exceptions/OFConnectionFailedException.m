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

#import "OFConnectionFailedException.h"
#import "OFString.h"

@implementation OFConnectionFailedException
+ (instancetype)exceptionWithHost: (OFString*)host
			     port: (uint16_t)port
			   socket: (id)socket
{
	return [[[self alloc] initWithHost: host
				      port: port
				    socket: socket] autorelease];
}

+ (instancetype)exceptionWithHost: (OFString*)host
			     port: (uint16_t)port
			   socket: (id)socket
			    errNo: (int)errNo
{
	return [[[self alloc] initWithHost: host
				      port: port
				    socket: socket
				     errNo: errNo] autorelease];
}

- init
{
	OF_INVALID_INIT_METHOD
}

- initWithHost: (OFString*)host
	  port: (uint16_t)port
	socket: (id)socket
{
	self = [super init];

	@try {
		_host = [host copy];
		_socket = [socket retain];
		_port = port;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithHost: (OFString*)host
	  port: (uint16_t)port
	socket: (id)socket
	 errNo: (int)errNo
{
	self = [super init];

	@try {
		_host = [host copy];
		_socket = [socket retain];
		_port = port;
		_errNo = errNo;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (void)dealloc
{
	[_host release];
	[_socket release];

	[super dealloc];
}

- (OFString*)description
{
	if (_errNo != 0)
		return [OFString stringWithFormat:
		    @"A connection to %@ on port %" @PRIu16 @" could not be "
		    @"established in socket of type %@: %@",
		    _host, _port, [_socket class], of_strerror(_errNo)];
	else
		return [OFString stringWithFormat:
		    @"A connection to %@ on port %" @PRIu16 @" could not be "
		    @"established in socket of type %@!",
		    _host, _port, [_socket class]];
}

- (OFString*)host
{
	OF_GETTER(_host, true)
}

- (uint16_t)port
{
	return _port;
}

- (id)socket
{
	OF_GETTER(_socket, true)
}

- (int)errNo
{
	return _errNo;
}
@end
