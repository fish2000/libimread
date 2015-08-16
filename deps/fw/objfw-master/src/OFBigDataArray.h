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

#import "OFDataArray.h"

OF_ASSUME_NONNULL_BEGIN

/*!
 * @class OFBigDataArray OFBigDataArray.h ObjFW/OFBigDataArray.h
 *
 * @brief A class for storing arbitrary big data in an array.
 *
 * The OFBigDataArray class is a class for storing arbitrary data in an array
 * and is designed to store large hunks of data. Therefore, it allocates
 * memory in pages rather than a chunk of memory for each item.
 */
@interface OFBigDataArray: OFDataArray
{
	size_t _size;
}
@end

OF_ASSUME_NONNULL_END
