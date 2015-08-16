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

/*!
 * @class OFUnsupportedVersionException \
 *	  OFUnsupportedVersionException.h ObjFW/OFUnsupportedVersionException.h
 *
 * @brief An exception indicating that the specified version of the format or
 *	  protocol is not supported.
 */
@interface OFUnsupportedVersionException: OFException
{
	OFString *_version;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, copy) OFString *version;
#endif

/*!
 * @brief Creates a new, autoreleased unsupported version exception.
 *
 * @param version The version which is unsupported
 * @return A new, autoreleased unsupported version exception
 */
+ (instancetype)exceptionWithVersion: (OFString*)version;

/*!
 * @brief Initializes an already allocated unsupported protocol exception.
 *
 * @param version The version which is unsupported
 * @return An initialized unsupported version exception
 */
- initWithVersion: (OFString*)version;

/*!
 * @brief Returns the version which is unsupported.
 *
 * @return The version which is unsupported
 */
- (OFString*)version;
@end
