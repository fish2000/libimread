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

#include <ctype.h>
#include <errno.h>
#include <string.h>

#import "OFHTTPClient.h"
#import "OFHTTPRequest.h"
#import "OFHTTPResponse.h"
#import "OFString.h"
#import "OFURL.h"
#import "OFTCPSocket.h"
#import "OFDictionary.h"
#import "OFDataArray.h"

#import "OFHTTPRequestFailedException.h"
#import "OFInvalidEncodingException.h"
#import "OFInvalidFormatException.h"
#import "OFInvalidServerReplyException.h"
#import "OFNotImplementedException.h"
#import "OFOutOfMemoryException.h"
#import "OFOutOfRangeException.h"
#import "OFReadFailedException.h"
#import "OFTruncatedDataException.h"
#import "OFUnsupportedProtocolException.h"
#import "OFUnsupportedVersionException.h"
#import "OFWriteFailedException.h"

static OF_INLINE void
normalizeKey(char *str_)
{
	uint8_t *str = (uint8_t*)str_;
	bool firstLetter = true;

	while (*str != '\0') {
		if (!isalnum(*str)) {
			firstLetter = true;
			str++;
			continue;
		}

		*str = (firstLetter ? toupper(*str) : tolower(*str));

		firstLetter = false;
		str++;
	}
}

@interface OFHTTPClientResponse: OFHTTPResponse
{
	OFTCPSocket *_socket;
	bool _hasContentLength, _chunked, _keepAlive, _atEndOfStream;
	size_t _toRead;
}

- initWithSocket: (OFTCPSocket*)socket;
- (void)OF_setKeepAlive: (bool)keepAlive;
@end

@implementation OFHTTPClientResponse
- initWithSocket: (OFTCPSocket*)socket
{
	self = [super init];

	_socket = [socket retain];

	return self;
}

- (void)OF_setKeepAlive: (bool)keepAlive
{
	_keepAlive = keepAlive;
}

- (void)dealloc
{
	[_socket release];

	[super dealloc];
}

- (void)setHeaders: (OFDictionary*)headers
{
	OFString *contentLength;

	[super setHeaders: headers];

	_chunked = [[headers objectForKey: @"Transfer-Encoding"]
	    isEqual: @"chunked"];

	contentLength = [headers objectForKey: @"Content-Length"];
	if (contentLength != nil) {
		_hasContentLength = true;

		@try {
			intmax_t toRead = [contentLength decimalValue];

			if (toRead > SIZE_MAX)
				@throw [OFOutOfRangeException exception];

			_toRead = (size_t)toRead;
		} @catch (OFInvalidFormatException *e) {
			@throw [OFInvalidServerReplyException exception];
		}
	}
}

- (size_t)lowlevelReadIntoBuffer: (void*)buffer
			  length: (size_t)length
{
	if (_atEndOfStream)
		@throw [OFReadFailedException exceptionWithObject: self
						  requestedLength: length
							    errNo: ENOTCONN];

	if (!_hasContentLength && !_chunked)
		return [_socket readIntoBuffer: buffer
					length: length];

	/* Content-Length */
	if (!_chunked) {
		size_t ret;

		if (_toRead == 0) {
			_atEndOfStream = true;

			if (!_keepAlive)
				[_socket close];

			return 0;
		}

		if (_toRead < length)
			ret = [_socket readIntoBuffer: buffer
					       length: _toRead];
		else
			ret = [_socket readIntoBuffer: buffer
					       length: length];

		_toRead -= ret;

		return ret;
	}

	/* Chunked */
	if (_toRead > 0) {
		if (length > _toRead)
			length = _toRead;

		length = [_socket readIntoBuffer: buffer
					  length: length];

		_toRead -= length;

		if (_toRead == 0)
			if ([[_socket readLine] length] > 0)
				@throw [OFInvalidServerReplyException
				    exception];

		return length;
	} else {
		void *pool = objc_autoreleasePoolPush();
		OFString *line;
		of_range_t range;

		@try {
			line = [_socket readLine];
		} @catch (OFInvalidEncodingException *e) {
			@throw [OFInvalidServerReplyException exception];
		}

		range = [line rangeOfString: @";"];
		if (range.location != OF_NOT_FOUND)
			line = [line substringWithRange:
			    of_range(0, range.location)];

		@try {
			uintmax_t toRead = [line hexadecimalValue];

			if (toRead > SIZE_MAX)
				@throw [OFOutOfRangeException exception];

			_toRead = (size_t)toRead;
		} @catch (OFInvalidFormatException *e) {
			@throw [OFInvalidServerReplyException exception];
		}

		if (_toRead == 0) {
			_atEndOfStream = true;

			if (_keepAlive) {
				@try {
					line = [_socket readLine];
				} @catch (OFInvalidEncodingException *e) {
					@throw [OFInvalidServerReplyException
					    exception];
				}

				if ([line length] > 0)
					@throw [OFInvalidServerReplyException
					    exception];
			} else
				[_socket close];
		}

		objc_autoreleasePoolPop(pool);

		return 0;
	}
}

- (bool)lowlevelIsAtEndOfStream
{
	if (!_hasContentLength && !_chunked)
		return [_socket isAtEndOfStream];

	return _atEndOfStream;
}

- (int)fileDescriptorForReading
{
	if (_socket == nil)
		return -1;

	return [_socket fileDescriptorForReading];
}

- (bool)hasDataInReadBuffer
{
	return ([super hasDataInReadBuffer] || [_socket hasDataInReadBuffer]);
}

- (void)close
{
	[_socket release];
	_socket = nil;
}
@end

@implementation OFHTTPClient
+ (instancetype)client
{
	return [[[self alloc] init] autorelease];
}

- (void)dealloc
{
	[self close];

	[super dealloc];
}

- (void)setDelegate: (id <OFHTTPClientDelegate>)delegate
{
	_delegate = delegate;
}

- (id <OFHTTPClientDelegate>)delegate
{
	return _delegate;
}

- (void)setInsecureRedirectsAllowed: (bool)allowed
{
	_insecureRedirectsAllowed = allowed;
}

- (bool)insecureRedirectsAllowed
{
	return _insecureRedirectsAllowed;
}

- (OFHTTPResponse*)performRequest: (OFHTTPRequest*)request
{
	return [self performRequest: request
			  redirects: 10];
}

- (OFTCPSocket*)OF_closeAndCreateSocketForRequest: (OFHTTPRequest*)request
{
	OFURL *URL = [request URL];
	OFTCPSocket *socket;

	[self close];

	if ([[URL scheme] isEqual: @"https"]) {
		if (of_tls_socket_class == Nil)
			@throw [OFUnsupportedProtocolException
			    exceptionWithURL: URL];

		socket = [[[of_tls_socket_class alloc] init]
		    autorelease];
	} else
		socket = [OFTCPSocket socket];

	if ([_delegate respondsToSelector:
	    @selector(client:didCreateSocket:request:)])
		[_delegate client: self
		  didCreateSocket: socket
			  request: request];

	[socket connectToHost: [URL host]
			 port: [URL port]];

	return socket;
}

- (OFHTTPResponse*)performRequest: (OFHTTPRequest*)request
			redirects: (size_t)redirects
{
	void *pool = objc_autoreleasePoolPush();
	OFURL *URL = [request URL];
	OFString *scheme = [URL scheme];
	of_http_request_method_t method = [request method];
	OFMutableString *requestString;
	OFString *user, *password;
	OFDictionary *headers = [request headers];
	OFDataArray *body = [request body];
	OFTCPSocket *socket;
	OFHTTPClientResponse *response;
	OFString *line, *version, *redirect, *connectionHeader;
	bool keepAlive;
	OFMutableDictionary *serverHeaders;
	OFEnumerator *keyEnumerator, *objectEnumerator;
	OFString *key, *object;
	int status;

	if (![scheme isEqual: @"http"] && ![scheme isEqual: @"https"])
		@throw [OFUnsupportedProtocolException exceptionWithURL: URL];

	/* Can we reuse the socket? */
	if (_socket != nil && [[_lastURL scheme] isEqual: scheme] &&
	    [[_lastURL host] isEqual: [URL host]] &&
	    [_lastURL port] == [URL port]) {
		/*
		 * Set _socket to nil, so that in case of an error it won't be
		 * reused. If everything is successfull, we set _socket again
		 * at the end.
		 */
		socket = [_socket autorelease];
		_socket = nil;

		[_lastURL release];
		_lastURL = nil;

		@try {
			if (!_lastWasHEAD) {
				/*
				 * Throw away content that has not been read
				 * yet.
				 */
				while (![_lastResponse isAtEndOfStream]) {
					char buffer[512];

					[_lastResponse readIntoBuffer: buffer
							       length: 512];
				}
			}
		} @finally {
			[_lastResponse release];
			_lastResponse = nil;
		}
	} else
		socket = [self OF_closeAndCreateSocketForRequest: request];

	/*
	 * As a work around for a bug with split packets in lighttpd when using
	 * HTTPS, we construct the complete request in a buffer string and then
	 * send it all at once.
	 */
	if ([URL query] != nil)
		requestString = [OFMutableString stringWithFormat:
		    @"%s /%@?%@ HTTP/%@\r\n",
		    of_http_request_method_to_string(method), [URL path],
		    [URL query], [request protocolVersionString]];
	else
		requestString = [OFMutableString stringWithFormat:
		    @"%s /%@ HTTP/%@\r\n",
		    of_http_request_method_to_string(method), [URL path],
		    [request protocolVersionString]];

	if (([scheme isEqual: @"http"] && [URL port] != 80) ||
	    ([scheme isEqual: @"https"] && [URL port] != 443))
		[requestString appendFormat: @"Host: %@:%d\r\n",
					     [URL host], [URL port]];
	else
		[requestString appendFormat: @"Host: %@\r\n", [URL host]];

	user = [URL user];
	password = [URL password];

	if ([user length] > 0 || [password length] > 0) {
		OFDataArray *authorization = [OFDataArray dataArray];

		[authorization addItems: [user UTF8String]
				  count: [user UTF8StringLength]];
		[authorization addItem: ":"];
		[authorization addItems: [password UTF8String]
				  count: [password UTF8StringLength]];

		[requestString appendFormat:
		    @"Authorization: Basic %@\r\n",
		    [authorization stringByBase64Encoding]];
	}

	if ([headers objectForKey: @"User-Agent"] == nil)
		[requestString appendString:
		    @"User-Agent: Something using ObjFW "
		    @"<https://webkeks.org/objfw>\r\n"];

	if (body != nil) {
		if ([headers objectForKey: @"Content-Length"] == nil)
			[requestString appendFormat:
			    @"Content-Length: %zd\r\n",
			    [body itemSize] * [body count]];

		if ([headers objectForKey: @"Content-Type"] == nil)
			[requestString appendString:
			    @"Content-Type: application/x-www-form-urlencoded; "
			    @"charset=UTF-8\r\n"];
	}

	keyEnumerator = [headers keyEnumerator];
	objectEnumerator = [headers objectEnumerator];

	while ((key = [keyEnumerator nextObject]) != nil &&
	    (object = [objectEnumerator nextObject]) != nil)
		[requestString appendFormat: @"%@: %@\r\n", key, object];

	if ([request protocolVersion].major == 1 &&
	    [request protocolVersion].minor == 0)
		[requestString appendString: @"Connection: keep-alive\r\n"];

	[requestString appendString: @"\r\n"];

	@try {
		[socket writeString: requestString];
	} @catch (OFWriteFailedException *e) {
		if ([e errNo] != ECONNRESET && [e errNo] != EPIPE)
			@throw e;

		/* Reconnect in case a keep-alive connection timed out */
		socket = [self OF_closeAndCreateSocketForRequest: request];
		[socket writeString: requestString];
	}

	if (body != nil)
		[socket writeBuffer: [body items]
			     length: [body count] * [body itemSize]];

	@try {
		line = [socket readLine];
	} @catch (OFInvalidEncodingException *e) {
		@throw [OFInvalidServerReplyException exception];
	}

	/*
	 * It's possible that the write succeeds on a connection that is
	 * keep-alive, but the connection has already been closed by the remote
	 * end due to a timeout. In this case, we need to reconnect.
	 */
	if (line == nil) {
		socket = [self OF_closeAndCreateSocketForRequest: request];
		[socket writeString: requestString];

		if (body != nil)
			[socket writeBuffer: [body items]
				     length: [body count] *
					     [body itemSize]];

		@try {
			line = [socket readLine];
		} @catch (OFInvalidEncodingException *e) {
			@throw [OFInvalidServerReplyException exception];
		}
	}

	if (![line hasPrefix: @"HTTP/"] || [line length] < 9 ||
	    [line characterAtIndex: 8] != ' ')
		@throw [OFInvalidServerReplyException exception];

	version = [line substringWithRange: of_range(5, 3)];
	if (![version isEqual: @"1.0"] && ![version isEqual: @"1.1"])
		@throw [OFUnsupportedVersionException
		    exceptionWithVersion: version];

	status = (int)[[line substringWithRange: of_range(9, 3)] decimalValue];

	serverHeaders = [OFMutableDictionary dictionary];

	for (;;) {
		OFString *key, *value;
		const char *lineC, *tmp;
		char *keyC;

		@try {
			line = [socket readLine];
		} @catch (OFInvalidEncodingException *e) {
			@throw [OFInvalidServerReplyException exception];
		}

		if (line == nil)
			@throw [OFInvalidServerReplyException exception];

		if ([line length] == 0)
			break;

		lineC = [line UTF8String];

		if ((tmp = strchr(lineC, ':')) == NULL)
			@throw [OFInvalidServerReplyException exception];

		if ((keyC = malloc(tmp - lineC + 1)) == NULL)
			@throw [OFOutOfMemoryException
			    exceptionWithRequestedSize: tmp - lineC + 1];

		memcpy(keyC, lineC, tmp - lineC);
		keyC[tmp - lineC] = '\0';
		normalizeKey(keyC);

		@try {
			key = [OFString stringWithUTF8StringNoCopy: keyC
						      freeWhenDone: true];
		} @catch (id e) {
			free(keyC);
			@throw e;
		}

		do {
			tmp++;
		} while (*tmp == ' ');

		value = [OFString stringWithUTF8String: tmp];

		[serverHeaders setObject: value
				  forKey: key];
	}

	[serverHeaders makeImmutable];

	if ([_delegate respondsToSelector:
	    @selector(client:didReceiveHeaders:statusCode:request:)])
		[_delegate     client: self
		    didReceiveHeaders: serverHeaders
			   statusCode: status
			      request: request];

	response = [[[OFHTTPClientResponse alloc] initWithSocket: socket]
	    autorelease];
	[response setProtocolVersionFromString: version];
	[response setStatusCode: status];
	[response setHeaders: serverHeaders];

	connectionHeader = [serverHeaders objectForKey: @"Connection"];
	if ([version isEqual: @"1.1"]) {
		if (connectionHeader != nil)
			keepAlive = ([connectionHeader caseInsensitiveCompare:
			    @"close"] != OF_ORDERED_SAME);
		else
			keepAlive = true;
	} else {
		if (connectionHeader != nil)
			keepAlive = ([connectionHeader caseInsensitiveCompare:
			    @"keep-alive"] == OF_ORDERED_SAME);
		else
			keepAlive = false;
	}

	if (keepAlive) {
		[response OF_setKeepAlive: true];

		_socket = [socket retain];
		_lastURL = [URL copy];
		_lastWasHEAD = (method == OF_HTTP_REQUEST_METHOD_HEAD);
		_lastResponse = [response retain];
	}

	/* FIXME: Case-insensitive check of redirect's scheme */
	if (redirects > 0 && (status == 301 || status == 302 ||
	    status == 303 || status == 307) &&
	    (redirect = [serverHeaders objectForKey: @"Location"]) != nil &&
	    (_insecureRedirectsAllowed || [scheme isEqual: @"http"] ||
	    [redirect hasPrefix: @"https://"])) {
		OFURL *newURL;
		bool follow;

		newURL = [OFURL URLWithString: redirect
				relativeToURL: URL];

		if ([_delegate respondsToSelector:
		    @selector(client:shouldFollowRedirect:statusCode:request:)])
			follow = [_delegate client: self
			      shouldFollowRedirect: newURL
					statusCode: status
					   request: request];
		else {
			/*
			 * 301, 302 and 307 should only redirect with user
			 * confirmation if the request method is not GET or
			 * HEAD. Asking the delegate and getting true returned
			 * is considered user confirmation.
			 */
			if (method == OF_HTTP_REQUEST_METHOD_GET ||
			    method == OF_HTTP_REQUEST_METHOD_HEAD)
				follow = true;
			/*
			 * 303 should always be redirected and converted to a
			 * GET request.
			 */
			else if (status == 303)
				follow = true;
			else
				follow = false;
		}

		if (follow) {
			OFHTTPRequest *newRequest;

			newRequest = [OFHTTPRequest requestWithURL: newURL];
			[newRequest setMethod: method];
			[newRequest setHeaders: headers];
			[newRequest setBody: body];

			/*
			 * 303 means the request should be converted to a GET
			 * request before redirection. This also means stripping
			 * the entity of the request.
			 */
			if (status == 303) {
				OFMutableDictionary *newHeaders;
				OFEnumerator *keyEnumerator, *objectEnumerator;
				id key, object;

				newHeaders = [OFMutableDictionary dictionary];
				keyEnumerator = [headers keyEnumerator];
				objectEnumerator = [headers objectEnumerator];
				while ((key = [keyEnumerator nextObject]) !=
				    nil &&
				    (object = [objectEnumerator nextObject]) !=
				    nil)
					if (![key hasPrefix: @"Content-"])
						[newHeaders setObject: object
							       forKey: key];

				[newRequest
				    setMethod: OF_HTTP_REQUEST_METHOD_GET];
				[newRequest setHeaders: newHeaders];
				[newRequest setBody: nil];
			}

			[newRequest retain];
			objc_autoreleasePoolPop(pool);
			[newRequest autorelease];

			return [self performRequest: newRequest
					  redirects: redirects - 1];
		}
	}

	[response retain];
	objc_autoreleasePoolPop(pool);
	[response autorelease];

	if (status / 100 != 2)
		@throw [OFHTTPRequestFailedException
		    exceptionWithRequest: request
				response: response];

	return response;
}

- (void)close
{
	[_socket close];
	[_socket release];
	_socket = nil;

	[_lastURL release];
	_lastURL = nil;

	[_lastResponse release];
	_lastResponse = nil;
}
@end
