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

#import "OFException.h"

@class OFStream;

/*!
 * @class OFSetOptionFailedException \
 *	  OFSetOptionFailedException.h ObjFW/OFSetOptionFailedException.h
 *
 * @brief An exception indicating that setting an option for a stream failed.
 */
@interface OFSetOptionFailedException: OFException
{
	OFStream *_stream;
	int _errNo;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, retain) OFStream *stream;
@property (readonly) int errNo;
#endif

/*!
 * @brief Creates a new, autoreleased set option failed exception.
 *
 * @param stream The stream for which the option could not be set
 * @param errNo The errno of the error that occurred
 * @return A new, autoreleased set option failed exception
 */
+ (instancetype)exceptionWithStream: (OFStream*)stream
			      errNo: (int)errNo;

/*!
 * @brief Initializes an already allocated set option failed exception.
 *
 * @param stream The stream for which the option could not be set
 * @param errNo The errno of the error that occurred
 * @return An initialized set option failed exception
 */
- initWithStream: (OFStream*)stream
	   errNo: (int)errNo;

/*!
 * @brief Returns the stream for which the option could not be set.
 *
 * @return The stream for which the option could not be set
 */
- (OFStream*)stream;

/*!
 * @brief Returns the errno of the error that occurred.
 *
 * @return The errno of the error that occurred
 */
- (int)errNo;
@end
