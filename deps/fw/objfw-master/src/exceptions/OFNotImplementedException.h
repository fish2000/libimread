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
 * @class OFNotImplementedException \
 *	  OFNotImplementedException.h ObjFW/OFNotImplementedException.h
 *
 * @brief An exception indicating that a method or part of it is not
 *        implemented.
 */
@interface OFNotImplementedException: OFException
{
	SEL _selector;
	id _object;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly) SEL selector;
@property (readonly, retain) id object;
#endif

/*!
 * @brief Creates a new, autoreleased not implemented exception.
 *
 * @param selector The selector which is not or not fully implemented
 * @param object The object which does not (fully) implement the selector
 * @return A new, autoreleased not implemented exception
 */
+ (instancetype)exceptionWithSelector: (SEL)selector
			       object: (id)object;

/*!
 * @brief Initializes an already allocated not implemented exception.
 *
 * @param selector The selector which is not or not fully implemented
 * @param object The object which does not (fully) implement the selector
 * @return An initialized not implemented exception
 */
- initWithSelector: (SEL)selector
	    object: (id)object;

/*!
 * @brief Returns the selector which is not or not fully implemented.
 *
 * @return The selector which is not or not fully implemented
 */
- (SEL)selector;

/*!
 * @brief Returns the object which does not (fully) implement the selector.
 *
 * @return The object which does not (fully) implement the selector
 */
- (id)object;
@end
