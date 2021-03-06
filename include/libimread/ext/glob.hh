/*
  Simple "glob" pattern matching.
  '*' matches zero or more of any character.
  '?' matches any single character.

  Example:

    if (glob::match("hello*", arg)) ...
*/

/*
 This software is distributed under the "Simplified BSD license":

 Copyright Michael Cook <michael@waxrat.com>. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef LIBIMREAD_INCLUDE_EXT_GLOB_HH_
#define LIBIMREAD_INCLUDE_EXT_GLOB_HH_

#include <experimental/string_view>
#include <functional>

namespace glob {
    
    using stringview_t = std::string_view;
    using glob_f = std::function<bool(stringview_t)>;
    
    bool  match(stringview_t pattern,
                stringview_t target);
    
    bool imatch(stringview_t pattern,
                stringview_t target);
    
    glob_f  matcher(stringview_t pattern);
    glob_f imatcher(stringview_t pattern);
    
}

#endif /// LIBIMREAD_INCLUDE_EXT_GLOB_HH_