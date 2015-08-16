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
#include <string.h>

#import "runtime.h"
#import "runtime-private.h"

#import "macros.h"

#ifdef OF_SELUID24
# define SEL_MAX 0xFFFFFF
# define SEL_SIZE 3
#else
# define SEL_MAX 0xFFFF
# define SEL_SIZE 2
#endif

static struct objc_hashtable *selectors = NULL;
static uint32_t selectors_cnt = 0;
static struct objc_sparsearray *selector_names = NULL;
static void **free_list = NULL;
static size_t free_list_cnt = 0;

void
objc_register_selector(struct objc_abi_selector *sel)
{
	struct objc_selector *rsel;
	const char *name;

	if (selectors_cnt > SEL_MAX)
		OBJC_ERROR("Out of selector slots!");

	if (selectors == NULL)
		selectors = objc_hashtable_new(
		    objc_hash_string, objc_equal_string, 2);
	else if ((rsel = objc_hashtable_get(selectors, sel->name)) != NULL) {
		((struct objc_selector*)sel)->uid = rsel->uid;
		return;
	}

	if (selector_names == NULL)
		selector_names = objc_sparsearray_new(SEL_SIZE);

	name = sel->name;
	rsel = (struct objc_selector*)sel;
	rsel->uid = selectors_cnt++;

	objc_hashtable_set(selectors, name, rsel);
	objc_sparsearray_set(selector_names, (uint32_t)rsel->uid, (void*)name);
}

SEL
sel_registerName(const char *name)
{
	const struct objc_abi_selector *rsel;
	struct objc_abi_selector *sel;

	objc_global_mutex_lock();

	if (selectors != NULL &&
	    (rsel = objc_hashtable_get(selectors, name)) != NULL) {
		objc_global_mutex_unlock();
		return (SEL)rsel;
	}

	if ((sel = malloc(sizeof(struct objc_abi_selector))) == NULL)
		OBJC_ERROR("Not enough memory to allocate selector!");

	if ((sel->name = of_strdup(name)) == NULL)
		OBJC_ERROR("Not enough memory to allocate selector!");

	sel->types = NULL;

	if ((free_list = realloc(free_list,
	    sizeof(void*) * (free_list_cnt + 2))) == NULL)
		OBJC_ERROR("Not enough memory to allocate selector!");

	free_list[free_list_cnt++] = sel;
	free_list[free_list_cnt++] = (char*)sel->name;

	objc_register_selector(sel);

	objc_global_mutex_unlock();
	return (SEL)sel;
}

void
objc_register_all_selectors(struct objc_abi_symtab *symtab)
{
	struct objc_abi_selector *sel;

	if (symtab->sel_refs == NULL)
		return;

	for (sel = symtab->sel_refs; sel->name != NULL; sel++)
		objc_register_selector(sel);
}

const char*
sel_getName(SEL sel)
{
	const char *ret;

	objc_global_mutex_lock();
	ret = objc_sparsearray_get(selector_names, (uint32_t)sel->uid);
	objc_global_mutex_unlock();

	return ret;
}

bool
sel_isEqual(SEL sel1, SEL sel2)
{
	return (sel1->uid == sel2->uid);
}

void
objc_unregister_all_selectors(void)
{
	objc_hashtable_free(selectors);
	objc_sparsearray_free(selector_names);

	if (free_list != NULL) {
		size_t i;

		for (i = 0; i < free_list_cnt; i++)
			free(free_list[i]);

		free(free_list);
	}

	selectors = NULL;
	selectors_cnt = 0;
	selector_names = NULL;
	free_list = NULL;
	free_list_cnt = 0;
}
