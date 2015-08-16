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

#import "OFString.h"

OF_ASSUME_NONNULL_BEGIN

#if !defined(OF_CONSTANT_STRING_M) && \
    defined(OF_APPLE_RUNTIME) && !defined(__OBJC2__)
# ifdef __cplusplus
extern "C" {
# endif
extern void *_OFConstantStringClassReference;
# ifdef __cplusplus
}
# endif
#endif

/*!
 * @class OFConstantString OFConstantString.h ObjFW/OFConstantString.h
 *
 * @brief A class for storing constant strings using the `@""` literal.
 */
@interface OFConstantString: OFString
{
	char *_cString;
	unsigned int _cStringLength;
}
@end

OF_ASSUME_NONNULL_END
