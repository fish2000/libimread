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

#include <iostream>

#include "array.h"
#include "dictionary.h"

using namespace CxxFW;

int
main()
{
	Dictionary<const char*, std::string> dict;
	dict["foo"] = "bar";
	dict["qux"] = "lol";

	std::cout << dict["foo"] << std::endl;
	std::cout << dict["qux"] << std::endl;

	dict.enumerate([] (const char *const &key, const std::string &value,
	  bool &stop) {
		std::cout << key << " = " << value << std::endl;
	});

	Array<std::string> array;
	array.append("foo");
	array.append("bar");
	array.append("baz");

	std::cout << array[0] << array[1] << array[2] << std::endl;

	array.enumerate([] (size_t idx, const std::string &obj, bool &stop) {
		std::cout << obj << std::endl;
	});

	return 0;
}
