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

#ifndef __CXXFW_ARRAY_H__
#define __CXXFW_ARRAY_H__

#include <functional>
#include <stdexcept>

namespace CxxFW {

template <typename T> class Array {
public:
	Array(): _data(nullptr) {}
	~Array();
	T& operator [](size_t idx);
	void enumerate(
	    std::function<void (size_t idx, const T &obj, bool &stop)> func);
	void append(const T &obj);

protected:
	T *_data;
	size_t _count, _mutations;
};

template <typename T> Array<T>::~Array()
{
	for (size_t i = 0; i < _count; i++)
		_data[i].~T();

	free(_data);
}

template <typename T> T&
Array<T>::operator [](size_t idx)
{
	if (idx >= _count)
		throw std::range_error("Index out of range");

	return _data[idx];
}

template <typename T> void
Array<T>::enumerate(
    std::function<void (size_t idx, const T &obj, bool &stop)> func)
{
	bool stop = false;

	for (size_t i = 0; i < _count && !stop; i++)
		func(i, _data[i], stop);
}

template <typename T> void
Array<T>::append(const T &obj)
{
	if (SIZE_MAX - _count < 1 || SIZE_MAX / sizeof(T) < _count + 1)
		throw std::overflow_error("Integer overflow");

	T *data = (T*)realloc(_data, (_count + 1) * sizeof(T));
	if (data == nullptr)
		throw std::bad_alloc();

	new(&data[_count]) T(obj);

	_data = data;
	_count++;
}

}

#endif
