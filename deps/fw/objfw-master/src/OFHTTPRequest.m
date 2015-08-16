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

#import "OFHTTPRequest.h"
#import "OFString.h"
#import "OFURL.h"
#import "OFDictionary.h"
#import "OFDataArray.h"
#import "OFArray.h"

#import "OFInvalidFormatException.h"
#import "OFOutOfRangeException.h"
#import "OFUnsupportedVersionException.h"

const char*
of_http_request_method_to_string(of_http_request_method_t method)
{
	switch (method) {
	case OF_HTTP_REQUEST_METHOD_OPTIONS:
		return "OPTIONS";
	case OF_HTTP_REQUEST_METHOD_GET:
		return "GET";
	case OF_HTTP_REQUEST_METHOD_HEAD:
		return "HEAD";
	case OF_HTTP_REQUEST_METHOD_POST:
		return "POST";
	case OF_HTTP_REQUEST_METHOD_PUT:
		return "PUT";
	case OF_HTTP_REQUEST_METHOD_DELETE:
		return "DELETE";
	case OF_HTTP_REQUEST_METHOD_TRACE:
		return "TRACE";
	case OF_HTTP_REQUEST_METHOD_CONNECT:
		return "CONNECT";
	}

	return NULL;
}

of_http_request_method_t
of_http_request_method_from_string(const char *string)
{
	if (strcmp(string, "OPTIONS") == 0)
		return OF_HTTP_REQUEST_METHOD_OPTIONS;
	if (strcmp(string, "GET") == 0)
		return OF_HTTP_REQUEST_METHOD_GET;
	if (strcmp(string, "HEAD") == 0)
		return OF_HTTP_REQUEST_METHOD_HEAD;
	if (strcmp(string, "POST") == 0)
		return OF_HTTP_REQUEST_METHOD_POST;
	if (strcmp(string, "PUT") == 0)
		return OF_HTTP_REQUEST_METHOD_PUT;
	if (strcmp(string, "DELETE") == 0)
		return OF_HTTP_REQUEST_METHOD_DELETE;
	if (strcmp(string, "TRACE") == 0)
		return OF_HTTP_REQUEST_METHOD_TRACE;
	if (strcmp(string, "CONNECT") == 0)
		return OF_HTTP_REQUEST_METHOD_CONNECT;

	@throw [OFInvalidFormatException exception];
}

@implementation OFHTTPRequest
+ (instancetype)request
{
	return [[[self alloc] init] autorelease];
}

+ (instancetype)requestWithURL: (OFURL*)URL
{
	return [[[self alloc] initWithURL: URL] autorelease];
}

- init
{
	self = [super init];

	_method = OF_HTTP_REQUEST_METHOD_GET;
	_protocolVersion.major = 1;
	_protocolVersion.minor = 1;

	return self;
}

- initWithURL: (OFURL*)URL
{
	self = [self init];

	@try {
		[self setURL: URL];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (void)dealloc
{
	[_URL release];
	[_headers release];
	[_body release];
	[_remoteAddress release];

	[super dealloc];
}

- copy
{
	OFHTTPRequest *copy = [[OFHTTPRequest alloc] init];

	@try {
		copy->_method = _method;
		copy->_protocolVersion = _protocolVersion;
		[copy setURL: _URL];
		[copy setHeaders: _headers];
		[copy setBody: _body];
		[copy setRemoteAddress: _remoteAddress];
	} @catch (id e) {
		[copy release];
		@throw e;
	}

	return copy;
}

- (bool)isEqual: (id)object
{
	OFHTTPRequest *request;

	if (![object isKindOfClass: [OFHTTPRequest class]])
		return false;

	request = object;

	if (request->_method != _method ||
	    request->_protocolVersion.major != _protocolVersion.major ||
	    request->_protocolVersion.minor != _protocolVersion.minor ||
	    ![request->_URL isEqual: _URL] ||
	    ![request->_headers isEqual: _headers] ||
	    ![request->_body isEqual: _body] ||
	    ![request->_remoteAddress isEqual: _remoteAddress])
		return false;

	return true;
}

- (uint32_t)hash
{
	uint32_t hash;

	OF_HASH_INIT(hash);

	OF_HASH_ADD(hash, _method);
	OF_HASH_ADD(hash, _protocolVersion.major);
	OF_HASH_ADD(hash, _protocolVersion.minor);
	OF_HASH_ADD_HASH(hash, [_URL hash]);
	OF_HASH_ADD_HASH(hash, [_headers hash]);
	OF_HASH_ADD_HASH(hash, [_body hash]);
	OF_HASH_ADD_HASH(hash, [_remoteAddress hash]);

	OF_HASH_FINALIZE(hash);

	return hash;
}

- (void)setURL: (OFURL*)URL
{
	OF_SETTER(_URL, URL, true, 1)
}

- (OFURL*)URL
{
	OF_GETTER(_URL, true)
}

- (void)setMethod: (of_http_request_method_t)method
{
	_method = method;
}

- (of_http_request_method_t)method
{
	return _method;
}

- (void)setProtocolVersion: (of_http_request_protocol_version_t)protocolVersion
{
	if (protocolVersion.major != 1 || protocolVersion.minor > 1)
		@throw [OFUnsupportedVersionException
		    exceptionWithVersion: [OFString stringWithFormat: @"%u.%u",
					      protocolVersion.major,
					      protocolVersion.minor]];

	_protocolVersion = protocolVersion;
}

- (of_http_request_protocol_version_t)protocolVersion
{
	return _protocolVersion;
}

- (void)setProtocolVersionFromString: (OFString*)string
{
	void *pool = objc_autoreleasePoolPush();
	OFArray *components = [string componentsSeparatedByString: @"."];
	intmax_t major, minor;
	of_http_request_protocol_version_t protocolVersion;

	if ([components count] != 2)
		@throw [OFInvalidFormatException exception];

	major = [[components firstObject] decimalValue];
	minor = [[components lastObject] decimalValue];

	if (major < 0 || major > UINT8_MAX || minor < 0 || minor > UINT8_MAX)
		@throw [OFOutOfRangeException exception];

	protocolVersion.major = (uint8_t)major;
	protocolVersion.minor = (uint8_t)minor;

	[self setProtocolVersion: protocolVersion];

	objc_autoreleasePoolPop(pool);
}

- (OFString*)protocolVersionString
{
	return [OFString stringWithFormat: @"%u.%u", _protocolVersion.major,
					   _protocolVersion.minor];
}

- (void)setHeaders: (OFDictionary*)headers
{
	OF_SETTER(_headers, headers, true, 1)
}

- (OFDictionary*)headers
{
	OF_GETTER(_headers, true)
}

- (void)setBody: (OFDataArray*)body
{
	OF_SETTER(_body, body, true, 0)
}

- (void)setBodyFromString: (OFString*)string
{
	[self setBodyFromString: string
		       encoding: OF_STRING_ENCODING_UTF_8];
}

- (void)setBodyFromString: (OFString*)string
		 encoding: (of_string_encoding_t)encoding
{
	void *pool = objc_autoreleasePoolPush();
	OFDataArray *body = [OFDataArray dataArray];

	[body addItems: [string cStringWithEncoding: encoding]
		 count: [string cStringLengthWithEncoding: encoding]];
	[self setBody: body];

	objc_autoreleasePoolPop(pool);
}

- (OFDataArray*)body
{
	OF_GETTER(_body, true)
}

- (void)setRemoteAddress: (OFString*)remoteAddress
{
	OF_SETTER(_remoteAddress, remoteAddress, true, 1)
}

- (OFString*)remoteAddress
{
	OF_GETTER(_remoteAddress, true)
}

- (OFString*)description
{
	void *pool = objc_autoreleasePoolPush();
	const char *method = of_http_request_method_to_string(_method);
	OFString *indentedHeaders, *indentedBody, *ret;

	indentedHeaders = [[_headers description]
	    stringByReplacingOccurrencesOfString: @"\n"
				      withString: @"\n\t"];
	indentedBody = [[_body description]
	    stringByReplacingOccurrencesOfString: @"\n"
				      withString: @"\n\t"];

	ret = [[OFString alloc] initWithFormat:
	    @"<%@:\n\tURL = %@\n"
	    @"\tMethod = %s\n"
	    @"\tHeaders = %@\n"
	    @"\tBody = %@\n"
	    @"\tRemote address = %@\n"
	    @">",
	    [self class], _URL, method, indentedHeaders, indentedBody,
	    _remoteAddress];

	objc_autoreleasePoolPop(pool);

	return [ret autorelease];
}
@end
