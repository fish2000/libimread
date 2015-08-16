/*
 * Copyright (c) 2013, Jonathan Schleifer <js@webkeks.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstring>
#include <string>

#include "hash.h"

#define HASH_INIT(hash) hash = 0;
#define HASH_ADD(hash, byte)			\
	{					\
		hash += (uint8_t)(byte);	\
		hash += (hash << 10);		\
		hash ^= (hash >> 6);		\
	}
#define HASH_FINALIZE(hash)		\
	{				\
		hash += (hash << 3);	\
		hash ^= (hash >> 11);	\
		hash += (hash << 15);	\
	}

template <> CxxFW::Hash
CxxFW::hash<const char*>(const char *const &key)
{
	CxxFW::Hash hash;
	size_t len = strlen(key);

	HASH_INIT(hash)

	for (size_t i = 0; i < len; i++)
		HASH_ADD(hash, key[i])

	HASH_FINALIZE(hash)

	return hash;
}

template <> CxxFW::Hash
CxxFW::hash<std::string>(const std::string &key)
{
	CxxFW::Hash hash;
	size_t len = key.length();

	HASH_INIT(hash)

	for (size_t i = 0; i < len; i++)
		HASH_ADD(hash, key[i])

	HASH_FINALIZE(hash)

	return hash;
}
