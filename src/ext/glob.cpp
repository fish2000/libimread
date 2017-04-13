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

#include <libimread/ext/glob.hh>
#include <cctype>

namespace glob {
    
    using stringview_t = std::experimental::string_view;
    
    namespace detail {
        
        template <typename equal_f>
        bool match(stringview_t pattern, stringview_t target, equal_f equal) {
            auto p = pattern.begin(); auto pe = pattern.end();
            auto q = target.begin();  auto qe = target.end();
            while (true) {
                if (p == pe) { return q == qe; }
                if (*p == '*') {
                    ++p;
                    for (auto backtracker = qe; backtracker >= q; --backtracker) {
                        if (match(stringview_t(p, pe - p),
                                  stringview_t(backtracker, qe - backtracker),
                                  equal)) { return true; }
                    }
                    break;
                }
                if (q == qe) { break; }
                if (*p != '?' && !equal(*p, *q)) { break; }
                ++p, ++q;
            }
            return false;
        }
    
    } /// namespace detail
    
    bool match(stringview_t pattern, stringview_t target) {
        return detail::match(pattern, target,
                          [](int a, int b) { return a == b; });
    }
    
    bool imatch(stringview_t pattern, stringview_t target) {
        return detail::match(pattern, target,
                          [](int a, int b) { return std::tolower(a) == std::tolower(b); });
    }

} /// namespace glob