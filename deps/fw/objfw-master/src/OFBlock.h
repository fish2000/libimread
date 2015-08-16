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

#import "OFObject.h"

#import "block.h"

OF_ASSUME_NONNULL_BEGIN

/*!
 * @class OFBlock OFBlock.h ObjFW/OFBlock.h
 *
 * @brief The class for all blocks, since all blocks are also objects.
 */
@interface OFBlock: OFObject
@end

@interface OFStackBlock: OFBlock
@end

@interface OFGlobalBlock: OFBlock
@end

@interface OFMallocBlock: OFBlock
@end

OF_ASSUME_NONNULL_END
