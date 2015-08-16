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
 * @class OFInvalidJSONException \
 *	  OFInvalidJSONException.h ObjFW/OFInvalidJSONException.h
 *
 * @brief An exception indicating a JSON representation is invalid.
 */
@interface OFInvalidJSONException: OFException
{
	OFString *_string;
	size_t _line;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, copy) OFString *string;
@property (readonly) size_t line;
#endif

/*!
 * @brief Creates a new, autoreleased invalid JSON exception.
 *
 * @param string The string containing the invalid JSON representation
 * @param line The line in which the parsing error encountered
 * @return A new, autoreleased invalid JSON exception
 */
+ (instancetype)exceptionWithString: (OFString*)string
			       line: (size_t)line;

/*!
 * @brief Initializes an already allocated invalid JSON exception.
 *
 * @param string The string containing the invalid JSON representation
 * @param line The line in which the parsing error encountered
 * @return An initialized invalid JSON exception
 */
- initWithString: (OFString*)string
	    line: (size_t)line;

/*!
 * @brief Returns the string containing the invalid JSON representation.
 *
 * @return The string containing the invalid JSON representation
 */
- (OFString*)string;

/*!
 * @brief Returns the line in which parsing the JSON representation failed.
 *
 * @return The line in which parsing the JSON representation failed
 */
- (size_t)line;
@end
