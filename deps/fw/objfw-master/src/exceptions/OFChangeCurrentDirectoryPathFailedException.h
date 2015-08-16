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
 * @class OFChangeCurrentDirectoryPathFailedException \
 *	  OFChangeCurrentDirectoryPathFailedException.h \
 *	  ObjFW/OFChangeCurrentDirectoryPathFailedException.h
 *
 * @brief An exception indicating that changing the current directory path
 *	  failed.
 */
@interface OFChangeCurrentDirectoryPathFailedException: OFException
{
	OFString *_path;
	int _errNo;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, copy) OFString *path;
@property (readonly) int errNo;
#endif

/*!
 * @brief Creates a new, autoreleased change current directory path failed
 *	  exception.
 *
 * @param path The path of the directory to which the current path could not be
 *	       changed
 * @param errNo The errno of the error that occurred
 * @return A new, autoreleased change current directory path failed exception
 */
+ (instancetype)exceptionWithPath: (OFString*)path
			    errNo: (int)errNo;

/*!
 * @brief Initializes an already allocated change directory failed exception.
 *
 * @param path The path of the directory to which the current path could not be
 *	       changed
 * @param errNo The errno of the error that occurred
 * @return An initialized change current directory path failed exception
 */
- initWithPath: (OFString*)path
	 errNo: (int)errNo;

/*!
 * @brief Returns the path of the directory to which the current path could not
 *	  be changed.
 *
 * @return The path of the directory to which the current path could not be
 *	   changed
 */
- (OFString*)path;

/*!
 * @brief Returns the errno of the error that occurred.
 *
 * @return The errno of the error that occurred
 */
- (int)errNo;
@end
