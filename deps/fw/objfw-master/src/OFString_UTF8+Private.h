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

#import "OFString_UTF8.h"

OF_ASSUME_NONNULL_BEGIN

@interface OFString_UTF8 (OF_PRIVATE_CATEGORY)
- (instancetype)OF_initWithUTF8String: (const char*)UTF8String
			       length: (size_t)UTF8StringLength
			      storage: (char*)storage;
@end

OF_ASSUME_NONNULL_END
