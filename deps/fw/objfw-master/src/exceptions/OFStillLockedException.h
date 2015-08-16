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
#import "OFLocking.h"

/*!
 * @class OFStillLockedException \
 *	  OFStillLockedException.h ObjFW/OFStillLockedException.h
 *
 * @brief An exception indicating that a lock is still locked.
 */
@interface OFStillLockedException: OFException
{
	id <OFLocking> _lock;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, retain) id <OFLocking> lock;
#endif

/*!
 * @brief Creates a new, autoreleased still locked exception.
 *
 * @param lock The lock which is still locked
 * @return A new, autoreleased still locked exception
 */
+ (instancetype)exceptionWithLock: (id <OFLocking>)lock;

/*!
 * @brief Initializes an already allocated still locked exception.
 *
 * @param lock The lock which is still locked
 * @return An initialized still locked exception
 */
- initWithLock: (id <OFLocking>)lock;

/*!
 * @brief Returns the lock which is still locked.
 *
 * @return The lock which is still locked
 */
- (id <OFLocking>)lock;
@end
