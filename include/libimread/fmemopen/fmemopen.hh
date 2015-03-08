//
// Copyright 2012 Jeff Verkoeyen
// Originally ported from https://github.com/ingenuitas/python-tesseract/blob/master/fmemopen.c
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef FMEMOPEN_H_
#define FMEMOPEN_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <functional>
#include <libimread/libimread.hpp>

/**
 * A BSD port of the fmemopen Linux method using funopen.
 *
 * man docs for fmemopen:
 * http://linux.die.net/man/3/fmemopen
 *
 * man docs for funopen:
 * https://developer.apple.com/library/mac/#documentation/Darwin/Reference/ManPages/man3/funopen.3.html
 *
 * This method is ported from ingenuitas' python-tesseract project.
 *
 * You must call fclose on the returned file pointer or memory will be leaked.
 *
 *      @param buf The data that will be used to back the FILE* methods. Must be at least
 *                 @c size bytes.
 *      @param size The size of the @c buf data.
 *      @param mode The permitted stream operation modes.
 *      @returns A pointer that can be used in the fread/fwrite/fseek/fclose family of methods.
 *               If a failure occurred NULL will be returned.
 */

namespace memory {

    FILE *fmemopen(void *buf, std::size_t size, const char *mode = "r");

    template <typename F>
    struct fcloser {
        constexpr fcloser() noexcept = default;
        template <typename U> fcloser(const fcloser<U>&) noexcept {};
        void operator()(F *filehandle) { if (filehandle) std::fclose(filehandle); }
    };

    using buffer = std::unique_ptr<typename std::decay<FILE>::type, fcloser<FILE>>;

    std::function<buffer(void*, std::size_t)> source = [](void *buf, std::size_t size) {
        return fmemopen(buf, size);
    };
    
    std::function<buffer(void*, std::size_t)> sink = [](void *buf, std::size_t size) {
        return fmemopen(buf, size, "w");
    };

}

#endif // #ifndef FMEMOPEN_H_