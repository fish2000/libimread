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
 * @class OFCreateDirectoryFailedException \
 *	  OFCreateDirectoryFailedException.h \
 *	  ObjFW/OFCreateDirectoryFailedException.h
 *
 * @brief An exception indicating a directory couldn't be created.
 */
@interface OFCreateDirectoryFailedException: OFException
{
	OFString *_path;
	int _errNo;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, copy) OFString *path;
@property (readonly) int errNo;
#endif

/*!
 * @brief Creates a new, autoreleased create directory failed exception.
 *
 * @param path A string with the path of the directory which could not be
 *	       created
 * @param errNo The errno of the error that occurred
 * @return A new, autoreleased create directory failed exception
 */
+ (instancetype)exceptionWithPath: (OFString*)path
			    errNo: (int)errNo;

/*!
 * @brief Initializes an already allocated create directory failed exception.
 *
 * @param path A string with the path of the directory which could not be
 *	       created
 * @param errNo The errno of the error that occurred
 * @return An initialized create directory failed exception
 */
- initWithPath: (OFString*)path
	 errNo: (int)errNo;

/*!
 * @brief Returns a string with the path of the directory which couldn't be
 *	  created.
 *
 * @return A string with the path of the directory which couldn't be created
 */
- (OFString*)path;

/*!
 * @brief Returns the errno of the error that occurred.
 *
 * @return The errno of the error that occurred
 */
- (int)errNo;
@end
