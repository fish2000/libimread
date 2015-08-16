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

#include "config.h"

#include <stdio.h>
#include <stdlib.h>

#import "runtime.h"
#import "runtime-private.h"

static struct objc_dtable_level2 *empty_level2 = NULL;
#ifdef OF_SELUID24
static struct objc_dtable_level3 *empty_level3 = NULL;
#endif

static void
init(void)
{
	uint_fast16_t i;

	empty_level2 = malloc(sizeof(struct objc_dtable_level2));
	if (empty_level2 == NULL)
		OBJC_ERROR("Not enough memory to allocate dtable!");

#ifdef OF_SELUID24
	empty_level3 = malloc(sizeof(struct objc_dtable_level3));
	if (empty_level3 == NULL)
		OBJC_ERROR("Not enough memory to allocate dtable!");
#endif

#ifdef OF_SELUID24
	for (i = 0; i < 256; i++) {
		empty_level2->buckets[i] = empty_level3;
		empty_level3->buckets[i] = (IMP)0;
	}
#else
	for (i = 0; i < 256; i++)
		empty_level2->buckets[i] = (IMP)0;
#endif
}

struct objc_dtable*
objc_dtable_new(void)
{
	struct objc_dtable *dtable;
	uint_fast16_t i;

#ifdef OF_SELUID24
	if (empty_level2 == NULL || empty_level3 == NULL)
		init();
#else
	if (empty_level2 == NULL)
		init();
#endif

	if ((dtable = malloc(sizeof(struct objc_dtable))) == NULL)
		OBJC_ERROR("Not enough memory to allocate dtable!");

	for (i = 0; i < 256; i++)
		dtable->buckets[i] = empty_level2;

	return dtable;
}

void
objc_dtable_copy(struct objc_dtable *dst, struct objc_dtable *src)
{
	uint_fast16_t i, j;
#ifdef OF_SELUID24
	uint_fast16_t k;
#endif
	uint32_t idx;

	for (i = 0; i < 256; i++) {
		if (src->buckets[i] == empty_level2)
			continue;

#ifdef OF_SELUID24
		for (j = 0; j < 256; j++) {
			if (src->buckets[i]->buckets[j] == empty_level3)
				continue;

			for (k = 0; k < 256; k++) {
				IMP obj;

				obj = src->buckets[i]->buckets[j]->buckets[k];

				if (obj == (IMP)0)
					continue;

				idx = (uint32_t)
				    (((uint32_t)i << 16) | (j << 8) | k);
				objc_dtable_set(dst, idx, obj);
			}
		}
#else
		for (j = 0; j < 256; j++) {
			IMP obj;

			obj = src->buckets[i]->buckets[j];

			if (obj == (IMP)0)
				continue;

			idx = (uint32_t)((i << 8) | j);
			objc_dtable_set(dst, idx, obj);
		}
#endif
	}
}

void
objc_dtable_set(struct objc_dtable *dtable, uint32_t idx, IMP obj)
{
#ifdef OF_SELUID24
	uint8_t i = idx >> 16;
	uint8_t j = idx >> 8;
	uint8_t k = idx;
#else
	uint8_t i = idx >> 8;
	uint8_t j = idx;
#endif

	if (dtable->buckets[i] == empty_level2) {
		struct objc_dtable_level2 *level2;
		uint_fast16_t l;

		level2 = malloc(sizeof(struct objc_dtable_level2));

		if (level2 == NULL)
			OBJC_ERROR("Not enough memory to insert into dtable!");

		for (l = 0; l < 256; l++)
#ifdef OF_SELUID24
			level2->buckets[l] = empty_level3;
#else
			level2->buckets[l] = (IMP)0;
#endif

		dtable->buckets[i] = level2;
	}

#ifdef OF_SELUID24
	if (dtable->buckets[i]->buckets[j] == empty_level3) {
		struct objc_dtable_level3 *level3;
		uint_fast16_t l;

		level3 = malloc(sizeof(struct objc_dtable_level3));

		if (level3 == NULL)
			OBJC_ERROR("Not enough memory to insert into dtable!");

		for (l = 0; l < 256; l++)
			level3->buckets[l] = (IMP)0;

		dtable->buckets[i]->buckets[j] = level3;
	}

	dtable->buckets[i]->buckets[j]->buckets[k] = obj;
#else
	dtable->buckets[i]->buckets[j] = obj;
#endif
}

void
objc_dtable_free(struct objc_dtable *dtable)
{
	uint_fast16_t i;
#ifdef OF_SELUID24
	uint_fast16_t j;
#endif

	for (i = 0; i < 256; i++) {
		if (dtable->buckets[i] == empty_level2)
			continue;

#ifdef OF_SELUID24
		for (j = 0; j < 256; j++)
			if (dtable->buckets[i]->buckets[j] != empty_level3)
				free(dtable->buckets[i]->buckets[j]);
#endif

		free(dtable->buckets[i]);
	}

	free(dtable);
}

void
objc_dtable_cleanup(void)
{
	if (empty_level2 != NULL)
		free(empty_level2);
#ifdef OF_SELUID24
	if (empty_level3 != NULL)
		free(empty_level3);
#endif

	empty_level2 = NULL;
#ifdef OF_SELUID24
	empty_level3 = NULL;
#endif
}
