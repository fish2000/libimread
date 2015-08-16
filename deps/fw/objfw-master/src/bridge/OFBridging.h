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

/*!
 * @protocol OFBridging OFBridging.h ObjFW-Bridge/OFBridging.h
 *
 * @brief A protocol implemented by classes supporting bridging ObjFW objects
 *	  to Foundation objects.
 */
@protocol OFBridging
/*!
 * @brief Returns an instance of a Foundation object corresponding to the
 *	  receiver.
 *
 * If possible, the original object is wrapped. If this is not possible, an
 * autoreleased copy is created.
 *
 * @return The receiver as Foundation object
 */
- (id)NSObject;
@end
