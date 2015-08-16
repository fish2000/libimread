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

/*
 * This is pretty much OFMapTable from ObjFW.
 */

#ifndef __CXXFW_DICTIONARY_H__
#define __CXXFW_DICTIONARY_H__

#include <cstring>
#include <functional>
#include <stdexcept>

#include "hash.h"

namespace CxxFW {

template <typename K, typename V> class Dictionary {
protected:
	static const size_t MIN_CAPACITY = 16;

	struct Bucket {
		Bucket(Hash hash, K key): hash(hash), key(key) {}

		Hash hash;
		K key;
		V value;
	};

public:
	Dictionary(size_t capacity = 0);
	~Dictionary(void);
	V& operator [](const K &key);
	size_t count(void);
	bool has_key(const V &key);
	bool contains(const V &value);
	void remove(const K &key);
	void enumerate(std::function<void(const K &key, const V &value,
	    bool &stop)> func);

protected:
	void resizeForCount(size_t count);

	size_t _count, _capacity, _mutations;
	Bucket **_buckets;
};

template <typename K, typename V> Dictionary<K, V>::Dictionary(size_t capacity):
    _count(0), _mutations(0), _buckets(nullptr)
{
	if (capacity > HASH_MAX / sizeof(*_buckets) ||
	    capacity > HASH_MAX / 8)
		throw std::overflow_error("Integer overflow");

	/* FIXME: Check that * 2 does not overflow! */
	for (_capacity = 1; _capacity < capacity; _capacity *= 2);
	if (capacity * 8 / _capacity >= 6)
		_capacity *= 2;

	if (_capacity < MIN_CAPACITY)
		_capacity = MIN_CAPACITY;

	_buckets = new Bucket*[_capacity];
	memset(_buckets, 0, _capacity * sizeof(*_buckets));
}

template <typename K, typename V> Dictionary<K, V>::~Dictionary()
{
	for (size_t i = 0; i < _capacity; i++)
		if (_buckets[i] != nullptr && _buckets[i] != (Bucket*)1)
			delete _buckets[i];

	delete[] _buckets;
}

template <typename K, typename V> V&
Dictionary<K, V>::operator [](const K &key)
{
	Hash keyHash = hash<K>(key);
	size_t i, last = _capacity;

	for (i = keyHash & (_capacity - 1); i < last &&
	    _buckets[i] != nullptr; i++) {
		if (_buckets[i] == (Bucket*)1)
			continue;

		if (_buckets[i]->key == key)
			return _buckets[i]->value;
	}

	/* In case the last bucket is already used */
	if (i >= last) {
		last = keyHash & (_capacity - 1);

		for (i = 0; i < last && _buckets[i] != nullptr; i++) {
			if (_buckets[i] == (Bucket*)1)
				continue;

			if (_buckets[i]->key == key)
				return _buckets[i]->value;
		}
	}

	resizeForCount(_count + 1);

	_mutations++;
	last = _capacity;

	for (i = keyHash & (_capacity - 1); i < last &&
	    _buckets[i] != nullptr && _buckets[i] != (Bucket*)1; i++);

	/* In case the last bucket is already used */
	if (i >= last) {
		last = keyHash & (_capacity - 1);

		for (i = 0; i < last && _buckets[i] != nullptr &&
		    _buckets[i] != (Bucket*)1; i++);
	}

	if (i >= last)
		throw std::overflow_error("Dictionary full");

	_buckets[i] = new Bucket(keyHash, key);
	_count++;

	return _buckets[i]->value;
}

template <typename K, typename V> size_t
Dictionary<K, V>::count(void)
{
	return _count;
}

template <typename K, typename V> bool
Dictionary<K, V>::has_key(const V &key)
{
	if (_count == 0)
		return false;

	for (size_t i = 0; i < _capacity; i++)
		if (_buckets[i] != nullptr && _buckets[i] != (Bucket*)1)
			if (_buckets[i]->key == key)
				return true;

	return false;
}

template <typename K, typename V> bool
Dictionary<K, V>::contains(const V &value)
{
	if (_count == 0)
		return false;

	for (size_t i = 0; i < _capacity; i++)
		if (_buckets[i] != nullptr && _buckets[i] != (Bucket*)1)
			if (_buckets[i]->value == value)
				return true;

	return false;
}

template <typename K, typename V> void
Dictionary<K, V>::remove(const K &key)
{
	Hash keyHash = hash<K>(key);
	size_t i, last = _capacity;

	for (i = keyHash & (_capacity - 1); i < last &&
	    _buckets[i] != nullptr; i++) {
		if (_buckets[i] == (Bucket*)1)
			continue;

		if (_buckets[i]->key == key) {
			_mutations++;

			delete _buckets[i];
			_buckets[i] = (Bucket*)1;

			_count--;
			resizeForCount(_count);

			return;
		}
	}

	if (i < last)
		return;

	/* In case the last bucket is already used */
	last = keyHash & (_capacity - 1);

	for (i = 0; i < last && _buckets[i] != nullptr; i++) {
		if (_buckets[i] == (Bucket*)1)
			continue;

		if (_buckets[i]->key == key) {
			_mutations++;

			delete _buckets[i];
			_buckets[i] = (Bucket*)1;

			_count--;
			resizeForCount(_count);

			return;
		}
	}
}

template <typename K, typename V> void
Dictionary<K, V>::enumerate(std::function<void(const K &key, const V &value,
    bool &stop)> func)
{
	size_t mutations = _mutations;
	bool stop = false;

	for (size_t i = 0; i < _capacity && !stop; i++) {
		if (_mutations != mutations)
			throw std::logic_error("Mutation during enumeration");

		if (_buckets[i] != nullptr && _buckets[i] != (Bucket*)1)
			func(_buckets[i]->key, _buckets[i]->value, stop);
	}
}

template <typename K, typename V> void
Dictionary<K, V>::resizeForCount(size_t count)
{
	Bucket **buckets;

	if (count > HASH_MAX || count > HASH_MAX / sizeof(*_buckets) ||
	    count > HASH_MAX / 8)
		throw std::overflow_error("Integer overflow");

	size_t fullness = count * 8 / _capacity;

	size_t capacity;
	/* FIXME: Check that * 2 does not overflow! */
	if (fullness >= 6)
		capacity = _capacity * 2;
	else if (fullness <= 1)
		capacity = _capacity / 2;
	else
		return;

	/*
	 * Don't downsize if we have an initial capacity or if we would fall
	 * below the minimum capacity.
	 */
	if ((capacity < _capacity && count > _count) || capacity < MIN_CAPACITY)
		return;

	buckets = new Bucket*[capacity];
	memset(buckets, 0, capacity * sizeof(*buckets));

	for (size_t i = 0; i < _capacity; i++) {
		if (_buckets[i] != NULL && _buckets[i] != (Bucket*)1) {
			size_t j, last = capacity;

			for (j = _buckets[i]->hash & (capacity - 1);
			    j < last && buckets[j] != nullptr; j++);

			/* In case the last bucket is already used */
			if (j >= last) {
				last = _buckets[i]->hash & (capacity - 1);

				for (j = 0; j < last && buckets[j] != nullptr;
				    j++);
			}

			buckets[j] = _buckets[i];
		}
	}

	delete _buckets;
	_buckets = buckets;
	_capacity = capacity;
}

}

#endif
