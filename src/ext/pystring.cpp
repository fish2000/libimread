///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2008-2010, Sony Pictures Imageworks Inc
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
// Neither the name of the organization Sony Pictures Imageworks nor the
// names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER
// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include <libimread/ext/pystring.hh>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>

namespace pystring {

#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS) || defined(_MSC_VER)
#ifndef WINDOWS
#define WINDOWS
#endif
#endif

    // This definition codes from configure.in in the python src.
    // Strictly speaking this limits us to str sizes of 2**31.
    // Should we wish to handle this limit, we could use an architecture
    // specific #defines and read from ssize_t (unistd.h) if the header exists.
    // But in the meantime, the use of int assures maximum arch compatibility.
    // This must also equal the size used in the end = MAX_32BIT_INT default arg.

    typedef int Py_ssize_t;

/* helper macro to fixup start/end slice values */
#define ADJUST_INDICES(start, end, len)                                                             \
    if (end > len) {                                                                                \
        end = len;                                                                                  \
    } else if (end < 0) {                                                                           \
        end += len;                                                                                 \
        if (end < 0) { end = 0; }                                                                   \
    }                                                                                               \
    if (start < 0) {                                                                                \
        start += len;                                                                               \
        if (start < 0) { start = 0; }                                                               \
    }

    namespace {
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        /// why doesn't the std::reverse work?
        ///
        void reverse_strings(std::vector<std::string>& result) {
            for (std::vector<std::string>::size_type i = 0; i < result.size() / 2; i++) {
                std::swap(result[i], result[result.size() - 1 - i]);
            }
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        ///
        ///
        void split_whitespace(std::string const& str, std::vector<std::string>& result,
                              int maxsplit) {
            std::string::size_type i, j, len = str.size();
            for (i = j = 0; i < len;) {
                while (i < len && std::isspace(str[i])) { i++; }
                j = i;
                while (i < len && !std::isspace(str[i])) { i++; }
                if (j < i) {
                    if (maxsplit-- <= 0) { break; }
                    result.push_back(str.substr(j, i - j));
                    while (i < len && std::isspace(str[i])) { i++; }
                    j = i;
                }
            }
            if (j < len) {
                result.push_back(str.substr(j, len - j));
            }
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        ///
        ///
        void rsplit_whitespace(std::string const& str, std::vector<std::string>& result,
                               int maxsplit) {
            std::string::size_type len = str.size();
            std::string::size_type i, j;
            for (i = j = len; i > 0;) {
                while (i > 0 && std::isspace(str[i - 1])) { i--; }
                j = i;
                while (i > 0 && !std::isspace(str[i - 1])) { i--; }
                if (j > i) {
                    if (maxsplit-- <= 0) { break; }
                    result.push_back(str.substr(i, j - i));
                    while (i > 0 && std::isspace(str[i - 1])) { i--; }
                    j = i;
                }
            }
            if (j > 0) {
                result.push_back(str.substr(0, j));
            }
            reverse_strings(result);
        }
        
    } // anonymous namespace
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    void split(std::string const& str, std::vector<std::string>& result, std::string const& sep,
               int maxsplit) {
        result.clear();
        if (maxsplit < 0) {
            maxsplit = MAX_32BIT_INT; // result.max_size();
        }
        if (sep.size() == 0) {
            split_whitespace(str, result, maxsplit);
            return;
        }
        std::string::size_type i, j, len = str.size(), n = sep.size();
        i = j = 0;
        while (i + n <= len) {
            if (str[i] == sep[0] && str.substr(i, n) == sep) {
                if (maxsplit-- <= 0) { break; }
                result.push_back(str.substr(j, i - j));
                i = j = i + n;
            } else {
                i++;
            }
        }\
        result.push_back(str.substr(j, len - j));
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    void rsplit(std::string const& str, std::vector<std::string>& result, std::string const& sep,
                int maxsplit) {
        if (maxsplit < 0) {
            split(str, result, sep, maxsplit);
            return;
        }
        result.clear();
        if (sep.size() == 0) {
            rsplit_whitespace(str, result, maxsplit);
            return;
        }
        Py_ssize_t i, j, len = (Py_ssize_t)str.size(), n = (Py_ssize_t)sep.size();
        i = j = len;
        while (i >= n) {
            if (str[i - 1] == sep[n - 1] && str.substr(i - n, n) == sep) {
                if (maxsplit-- <= 0) { break; }
                result.push_back(str.substr(i, j - i));
                i = j = i - n;
            } else {
                i--;
            }
        }
        result.push_back(str.substr(0, j));
        reverse_strings(result);
    }

//////////////////////////////////////////////////////////////////////////////////////////////
///
///
#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2

    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string do_strip(std::string const& str, int striptype, std::string const& chars) {
        Py_ssize_t len = (Py_ssize_t)str.size(), i, j, charslen = (Py_ssize_t)chars.size();
        if (charslen == 0) {
            i = 0;
            if (striptype != RIGHTSTRIP) {
                while (i < len && std::isspace(str[i])) { i++; }
            }
            j = len;
            if (striptype != LEFTSTRIP) {
                do {
                    j--;
                } while (j >= i && std::isspace(str[j]));
                j++;
            }
        } else {
            const char* sep = chars.c_str();
            i = 0;
            if (striptype != RIGHTSTRIP) {
                while (i < len && std::memchr(sep, str[i], charslen)) { i++; }
            }
            j = len;
            if (striptype != LEFTSTRIP) {
                do {
                    j--;
                } while (j >= i && std::memchr(sep, str[j], charslen));
                j++;
            }
        }
        
        if (i == 0 && j == len) {
            return str;
        } else {
            return str.substr(i, j - i);
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    void partition(std::string const& str, std::string const& sep,
                   std::vector<std::string>& result) {
        result.resize(3);
        int idx = find(str, sep);
        if (idx < 0) {
            result[0] = str;
            result[1] = "";
            result[2] = "";
        } else {
            result[0] = str.substr(0, idx);
            result[1] = sep;
            result[2] = str.substr(idx + sep.size(), str.size());
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    void rpartition(std::string const& str, std::string const& sep,
                    std::vector<std::string>& result) {
        result.resize(3);
        int idx = rfind(str, sep);
        if (idx < 0) {
            result[0] = "";
            result[1] = "";
            result[2] = str;
        } else {
            result[0] = str.substr(0, idx);
            result[1] = sep;
            result[2] = str.substr(idx + sep.size(), str.size());
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string strip(std::string const& str, std::string const& chars) {
        return do_strip(str, BOTHSTRIP, chars);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string lstrip(std::string const& str, std::string const& chars) {
        return do_strip(str, LEFTSTRIP, chars);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string rstrip(std::string const& str, std::string const& chars) {
        return do_strip(str, RIGHTSTRIP, chars);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string join(std::string const& str, const std::vector<std::string>& seq) {
        std::vector<std::string>::size_type seqlen = seq.size(), i;
        if (seqlen == 0) { return ""; }
        if (seqlen == 1) { return seq[0]; }
        std::string result(seq[0]);
        for (i = 1; i < seqlen; ++i) {
            result += str + seq[i];
        }
        return result;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    
    namespace {
        /* Matches the end (direction >= 0) or start (direction < 0) of self
         * against substr, using the start and end arguments. Returns
         * -1 on error, 0 if not found and 1 if found.
         */
        
        int _string_tailmatch(std::string const& self, std::string const& substr, Py_ssize_t start,
                              Py_ssize_t end, int direction) {
            Py_ssize_t len  = (Py_ssize_t)self.size();
            Py_ssize_t slen = (Py_ssize_t)substr.size();
            const char* sub = substr.c_str();
            const char* str = self.c_str();
            ADJUST_INDICES(start, end, len);
            if (direction < 0) {
                // startswith
                if (start + slen > len) { return 0; }
            } else {
                // endswith
                if (end - start < slen || start > len) { return 0; }
                if (end - slen > start) { start = end - slen; }
            }
            if (end - start >= slen) { return (!std::memcmp(str + start, sub, slen)); }
            return 0;
        }
    } // namespace
    
    bool endswith(std::string const& str, std::string const& suffix, int start, int end) {
        int result = _string_tailmatch(str, suffix, (Py_ssize_t)start, (Py_ssize_t)end, +1);
        // if (result == -1) // TODO: Error condition
        return static_cast<bool>(result);
    }
    
    bool startswith(std::string const& str, std::string const& prefix, int start, int end) {
        int result = _string_tailmatch(str, prefix, (Py_ssize_t)start, (Py_ssize_t)end, -1);
        // if (result == -1) // TODO: Error condition
        return static_cast<bool>(result);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    
    bool isalnum(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::isalnum(str[0]); }
        for (i = 0; i < len; ++i) {
            if (!std::isalnum(str[i])) { return false; }
        }
        return true;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    bool isalpha(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::isalpha((int)str[0]); }
        for (i = 0; i < len; ++i) {
            if (!std::isalpha((int)str[i])) { return false; }
        }
        return true;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    bool isdigit(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::isdigit(str[0]); }
        for (i = 0; i < len; ++i) {
            if (!std::isdigit(str[i])) { return false; }
        }
        return true;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    bool islower(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::islower(str[0]); }
        for (i = 0; i < len; ++i) {
            if (!std::islower(str[i])) { return false; }
        }
        return true;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    bool isspace(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::isspace(str[0]); }
        for (i = 0; i < len; ++i) {
            if (!std::isspace(str[i])) { return false; }
        }
        return true;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    bool istitle(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::isupper(str[0]); }
        bool cased = false, previous_is_cased = false;
        for (i = 0; i < len; ++i) {
            if (std::isupper(str[i])) {
                if (previous_is_cased) { return false; }
                previous_is_cased = true;
                cased             = true;
            } else if (std::islower(str[i])) {
                if (!previous_is_cased) { return false; }
                previous_is_cased = true;
                cased             = true;
            } else {
                previous_is_cased = false;
            }
        }
        return cased;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    bool isupper(std::string const& str) {
        std::string::size_type len = str.size(), i;
        if (len == 0) { return false; }
        if (len == 1) { return std::isupper(str[0]); }
        for (i = 0; i < len; ++i) {
            if (!std::isupper(str[i])) { return false; }
        }
        return true;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string capitalize(std::string const& str) {
        std::string            s(str);
        std::string::size_type len = s.size(), i;
        if (len > 0) {
            if (std::islower(s[0])) { s[0] = (char)std::toupper(s[0]); }
        }
        for (i = 1; i < len; ++i) {
            if (std::isupper(s[i])) { s[i] = (char)std::tolower(s[i]); }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string lower(std::string const& str) {
        std::string            s(str);
        std::string::size_type len = s.size(), i;
        for (i = 0; i < len; ++i) {
            if (std::isupper(s[i])) { s[i] = (char)std::tolower(s[i]); }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string upper(std::string const& str) {
        std::string            s(str);
        std::string::size_type len = s.size(), i;
        for (i = 0; i < len; ++i) {
            if (std::islower(s[i])) { s[i] = (char)std::toupper(s[i]); }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string swapcase(std::string const& str) {
        std::string            s(str);
        std::string::size_type len = s.size(), i;
        for (i = 0; i < len; ++i) {
            if (std::islower(s[i])) { s[i] = (char)std::toupper(s[i]); }
            else if (std::isupper(s[i])) { s[i] = (char)std::tolower(s[i]); }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string title(std::string const& str) {
        std::string            s(str);
        std::string::size_type len               = s.size(), i;
        bool                   previous_is_cased = false;
        for (i = 0; i < len; ++i) {
            int c = s[i];
            if (std::islower(c)) {
                if (!previous_is_cased) { s[i] = (char)std::toupper(c); }
                previous_is_cased = true;
            } else if (std::isupper(c)) {
                if (previous_is_cased) { s[i] = (char)std::tolower(c); }
                previous_is_cased = true;
            } else {
                previous_is_cased = false;
            }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string translate(std::string const& str, std::string const& table,
                          std::string const& deletechars) {
        std::string            s;
        std::string::size_type len = str.size(), dellen = deletechars.size();
        
        if (table.size() != 256) {
            // TODO : raise exception instead
            return str;
        }
        // if nothing is deleted, use faster code
        if (dellen == 0) {
            s = str;
            for (std::string::size_type i = 0; i < len; ++i) {
                s[i] = table[s[i]];
            }
            return s;
        }
        int trans_table[256];
        for (int i = 0; i < 256; i++) {
            trans_table[i] = table[i];
        }
        for (std::string::size_type i = 0; i < dellen; i++) {
            trans_table[(int)deletechars[i]] = -1;
        }
        for (std::string::size_type i = 0; i < len; ++i) {
            if (trans_table[(int)str[i]] != -1) {
                s += table[str[i]];
            }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string zfill(std::string const& str, int width) {
        int len = (int)str.size();
        if (len >= width) {
            return str;
        }
        std::string s(str);
        int fill = width - len;
        s = std::string(fill, '0') + s;
        if (s[fill] == '+' || s[fill] == '-') {
            s[0]    = s[fill];
            s[fill] = '0';
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string ljust(std::string const& str, int width) {
        std::string::size_type len = str.size();
        if (((int)len) >= width) { return str; }
        return str + std::string(width - len, ' ');
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string rjust(std::string const& str, int width) {
        std::string::size_type len = str.size();
        if (((int)len) >= width) { return str; }
        return std::string(width - len, ' ') + str;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string center(std::string const& str, int width) {
        int len = (int)str.size();
        int marg, left;
        if (len >= width) { return str; }
        marg = width - len;
        left = marg / 2 + (marg & width & 1);
        return std::string(left, ' ') + str + std::string(marg - left, ' ');
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string slice(std::string const& str, int start, int end) {
        ADJUST_INDICES(start, end, (int)str.size());
        if (start >= end) { return ""; }
        return str.substr(start, end - start);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    int find(std::string const& str, std::string const& sub, int start, int end) {
        ADJUST_INDICES(start, end, (int)str.size());
        std::string::size_type result = str.find(sub, start);
        
        // If we cannot find the string, or if the end-point of our found substring is past
        // the allowed end limit, return that it can't be found.
        if (result == std::string::npos || (result + sub.size() > (std::string::size_type)end)) {
            return -1;
        }
        return (int)result;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    int index(std::string const& str, std::string const& sub, int start, int end) {
        return find(str, sub, start, end);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    int rfind(std::string const& str, std::string const& sub, int start, int end) {
        ADJUST_INDICES(start, end, (int)str.size());
        std::string::size_type result = str.rfind(sub, end);
        if (result == std::string::npos || result < (std::string::size_type)start ||
            (result + sub.size() > (std::string::size_type)end)) { return -1; }
        return (int)result;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    int rindex(std::string const& str, std::string const& sub, int start, int end) {
        return rfind(str, sub, start, end);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string expandtabs(std::string const& str, int tabsize) {
        std::string s(str);
        std::string::size_type len = str.size(), i = 0;
        int                    offset = 0;
        int j = 0;
        
        for (i = 0; i < len; ++i) {
            if (str[i] == '\t') {
                if (tabsize > 0) {
                    int fillsize = tabsize - (j % tabsize);
                    j += fillsize;
                    s.replace(i + offset, 1, std::string(fillsize, ' '));
                    offset += fillsize - 1;
                } else {
                    s.replace(i + offset, 1, "");
                    offset -= 1;
                }
            } else {
                j++;
                if (str[i] == '\n' || str[i] == '\r') { j = 0; }
            }
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    int count(std::string const& str, std::string const& substr, int start, int end) {
        int nummatches = 0;
        int cursor     = start;
        
        while (1) {
            cursor = find(str, substr, cursor, end);
            if (cursor < 0) { break; }
            cursor += (int)substr.size();
            nummatches += 1;
        }
        return nummatches;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    
    std::string replace(std::string const& str, std::string const& oldstr,
                        std::string const& newstr, int count) {
        int         sofar  = 0;
        int         cursor = 0;
        std::string s(str);
        std::string::size_type oldlen = oldstr.size(), newlen = newstr.size();
        cursor = find(s, oldstr, cursor);
        
        while (cursor != -1 && cursor <= (int)s.size()) {
            if (count > -1 && sofar >= count) { break; }
            s.replace(cursor, oldlen, newstr);
            cursor += (int)newlen;
            if (oldlen != 0) {
                cursor = find(s, oldstr, cursor);
            } else {
                ++cursor;
            }
            ++sofar;
        }
        return s;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    void splitlines(std::string const& str, std::vector<std::string>& result, bool keepends) {
        result.clear();
        std::string::size_type len = str.size(), i, j, eol;
        
        for (i = j = 0; i < len;) {
            while (i < len && str[i] != '\n' && str[i] != '\r') { i++; }
            eol = i;
            if (i < len) {
                if (str[i] == '\r' && i + 1 < len && str[i + 1] == '\n') {
                    i += 2;
                } else {
                    i++;
                }
                if (keepends) { eol = i; }
            }
            result.push_back(str.substr(j, eol - j));
            j = i;
        }
        if (j < len) {
            result.push_back(str.substr(j, len - j));
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    std::string mul(std::string const& str, int n) {
        // Early exits
        if (n <= 0) { return ""; }
        if (n == 1) { return str; }
        std::ostringstream os;
        for (int i = 0; i < n; ++i) {
            os << str;
        }
        return os.str();
    }
    
    namespace os {
        
        namespace path {
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            /// These functions are C++ ports of the python2.6 versions of os.path,
            /// and come from genericpath.py, ntpath.py, posixpath.py
            
            /// Split a pathname into drive and path specifiers.
            /// Returns drivespec, pathspec. Either part may be empty.
            void splitdrive_nt(std::string& drivespec, std::string& pathspec,
                               std::string const& p) {
                if (pystring::slice(p, 1, 2) == ":") {
                    std::string path = p; // In case drivespec == p
                    drivespec        = pystring::slice(path, 0, 2);
                    pathspec         = pystring::slice(path, 2);
                } else {
                    drivespec = "";
                    pathspec  = p;
                }
            }
            
            // On Posix, drive is always empty
            void splitdrive_posix(std::string& drivespec, std::string& pathspec,
                                  std::string const& path) {
                drivespec = "";
                pathspec  = path;
            }
            
            void splitdrive(std::string& drivespec, std::string& pathspec,
                            std::string const& path) {
#ifdef WINDOWS
                return splitdrive_nt(drivespec, pathspec, path);
#else
                return splitdrive_posix(drivespec, pathspec, path);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            // Test whether a path is absolute
            // In windows, if the character to the right of the colon
            // is a forward or backslash it's absolute.
            bool isabs_nt(std::string const& path) {
                std::string drivespec, pathspec;
                splitdrive_nt(drivespec, pathspec, path);
                if (pathspec.empty()) { return false; }
                return ((pathspec[0] == '/') || (pathspec[0] == '\\'));
            }
            
            bool isabs_posix(std::string const& s) { return pystring::startswith(s, "/"); }
            
            bool isabs(std::string const& path) {
#ifdef WINDOWS
                return isabs_nt(path);
#else
                return isabs_posix(path);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            std::string abspath_nt(std::string const& path, std::string const& cwd) {
                std::string p = path;
                if (!isabs_nt(p)) { p = join_nt(cwd, p); }
                return normpath_nt(p);
            }
            
            std::string abspath_posix(std::string const& path, std::string const& cwd) {
                std::string p = path;
                if (!isabs_posix(p)) { p = join_posix(cwd, p); }
                return normpath_posix(p);
            }
            
            std::string abspath(std::string const& path, std::string const& cwd) {
#ifdef WINDOWS
                return abspath_nt(path, cwd);
#else
                return abspath_posix(path, cwd);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            std::string join_nt(const std::vector<std::string>& paths) {
                if (paths.empty()) { return ""; }
                if (paths.size() == 1) { return paths[0]; }
                std::string path = paths[0];
                
                for (unsigned int i = 1; i < paths.size(); ++i) {
                    std::string b = paths[i];
                    bool b_nts = false;
                    if (path.empty()) {
                        b_nts = true;
                    } else if (isabs_nt(b)) {
                        // This probably wipes out path so far.  However, it's more
                        // complicated if path begins with a drive letter:
                        //     1. join('c:', '/a') == 'c:/a'
                        //     2. join('c:/', '/a') == 'c:/a'
                        // But
                        //     3. join('c:/a', '/b') == '/b'
                        //     4. join('c:', 'd:/') = 'd:/'
                        //     5. join('c:/', 'd:/') = 'd:/'
                        if ((pystring::slice(path, 1, 2) != ":") ||
                            (pystring::slice(b, 1, 2) == ":")) {
                            // Path doesnt start with a drive letter
                            b_nts = true;
                        } else if ((path.size() > 3) ||
                                 ((path.size() == 3) && !pystring::endswith(path, "/") &&
                                  !pystring::endswith(path, "\\"))) {
                            // Else path has a drive letter, and b doesn't but is absolute.
                            b_nts = true;
                        }
                    }
                    if (b_nts) {
                        path = b;
                    } else {
                        // Join, and ensure there's a separator.
                        // assert len(path) > 0
                        if (pystring::endswith(path, "/") || pystring::endswith(path, "\\")) {
                            if (pystring::startswith(b, "/") || pystring::startswith(b, "\\")) {
                                path += pystring::slice(b, 1);
                            } else {
                                path += b;
                            }
                        } else if (pystring::endswith(path, ":")) {
                            path += b;
                        } else if (!b.empty()) {
                            if (pystring::startswith(b, "/") || pystring::startswith(b, "\\")) {
                                path += b;
                            } else {
                                path += "\\" + b;
                            }
                        } else {
                            // path is not empty and does not end with a backslash,
                            // but b is empty; since, e.g., split('a/') produces
                            // ('a', ''), it's best if join() adds a backslash in
                            // this case.
                            path += "\\";
                        }
                    }
                }
                return path;
            }
            
            // Join two or more pathname components, inserting "\\" as needed.
            std::string join_nt(std::string const& a, std::string const& b) {
                std::vector<std::string> paths(2);
                paths[0] = a;
                paths[1] = b;
                return join_nt(paths);
            }
            
            // Join pathnames.
            // If any component is an absolute path, all previous path components
            // will be discarded.
            // Ignore the previous parts if a part is absolute.
            // Insert a '/' unless the first part is empty or already ends in '/'.
            
            std::string join_posix(const std::vector<std::string>& paths) {
                if (paths.empty()) { return ""; }
                if (paths.size() == 1) { return paths[0]; }
                std::string path = paths[0];
                
                for (unsigned int i = 1; i < paths.size(); ++i) {
                    std::string b = paths[i];
                    if (pystring::startswith(b, "/")) {
                        path = b;
                    } else if (path.empty() || pystring::endswith(path, "/")) {
                        path += b;
                    } else {
                        path += "/" + b;
                    }
                }
                return path;
            }
            
            std::string join_posix(std::string const& a, std::string const& b) {
                std::vector<std::string> paths(2);
                paths[0] = a;
                paths[1] = b;
                return join_posix(paths);
            }
            
            std::string join(std::string const& path1, std::string const& path2) {
#ifdef WINDOWS
                return join_nt(path1, path2);
#else
                return join_posix(path1, path2);
#endif
            }
            
            std::string join(const std::vector<std::string>& paths) {
#ifdef WINDOWS
                return join_nt(paths);
#else
                return join_posix(paths);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            // Split a pathname.
            // Return (head, tail) where tail is everything after the final slash.
            // Either part may be empty
            
            void split_nt(std::string& head, std::string& tail, std::string const& path) {
                std::string d, p;
                splitdrive_nt(d, p, path);
                
                // set i to index beyond p's last slash
                int i = (int)p.size();
                
                while (i > 0 && (p[i - 1] != '\\') && (p[i - 1] != '/')) {
                    i = i - 1;
                }
                head = pystring::slice(p, 0, i);
                tail = pystring::slice(p, i); // now tail has no slashes
                
                // remove trailing slashes from head, unless it's all slashes
                std::string head2 = head;
                while (!head2.empty() && ((pystring::slice(head2, -1) == "/") ||
                                          (pystring::slice(head2, -1) == "\\"))) {
                    head2 = pystring::slice(head, 0, -1);
                }
                if (!head2.empty()) { head = head2; }
                head = d + head;
            }
            
            // Split a path in head (everything up to the last '/') and tail (the
            // rest).  If the path ends in '/', tail will be empty.  If there is no
            // '/' in the path, head  will be empty.
            // Trailing '/'es are stripped from head unless it is the root.

            void split_posix(std::string& head, std::string& tail, std::string const& p) {
                int i = pystring::rfind(p, "/") + 1;
                
                head = pystring::slice(p, 0, i);
                tail = pystring::slice(p, i);
                
                if (!head.empty() && (head != pystring::mul("/", (int)head.size()))) {
                    head = pystring::rstrip(head, "/");
                }
            }
            
            void split(std::string& head, std::string& tail, std::string const& path) {
#ifdef WINDOWS
                return split_nt(head, tail, path);
#else
                return split_posix(head, tail, path);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            std::string basename_nt(std::string const& path) {
                std::string head, tail;
                split_nt(head, tail, path);
                return tail;
            }
            
            std::string basename_posix(std::string const& path) {
                std::string head, tail;
                split_posix(head, tail, path);
                return tail;
            }
            
            std::string basename(std::string const& path) {
#ifdef WINDOWS
                return basename_nt(path);
#else
                return basename_posix(path);
#endif
            }
            
            std::string dirname_nt(std::string const& path) {
                std::string head, tail;
                split_nt(head, tail, path);
                return head;
            }
            
            std::string dirname_posix(std::string const& path) {
                std::string head, tail;
                split_posix(head, tail, path);
                return head;
            }
            
            std::string dirname(std::string const& path) {
#ifdef WINDOWS
                return dirname_nt(path);
#else
                return dirname_posix(path);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            // Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A\B.
            std::string normpath_nt(std::string const& p) {
                std::string path = p;
                path             = pystring::replace(path, "/", "\\");
                std::string prefix;
                splitdrive_nt(prefix, path, path);
                
                // We need to be careful here. If the prefix is empty, and the path starts
                // with a backslash, it could either be an absolute path on the current
                // drive (\dir1\dir2\file) or a UNC filename (\\server\mount\dir1\file). It
                // is therefore imperative NOT to collapse multiple backslashes blindly in
                // that case.
                // The code below preserves multiple backslashes when there is no drive
                // letter. This means that the invalid filename \\\a\b is preserved
                // unchanged, where a\\\b is normalised to a\b. It's not clear that there
                // is any better behaviour for such edge cases.
                if (prefix.empty()) {
                    // No drive letter - preserve initial backslashes
                    while (pystring::slice(path, 0, 1) == "\\") {
                        prefix = prefix + "\\";
                        path   = pystring::slice(path, 1);
                    }
                } else {
                    // We have a drive letter - collapse initial backslashes
                    if (pystring::startswith(path, "\\")) {
                        prefix = prefix + "\\";
                        path   = pystring::lstrip(path, "\\");
                    }
                }
                
                std::vector<std::string> comps;
                pystring::split(path, comps, "\\");
                int i = 0;
                
                while (i < (int)comps.size()) {
                    if (comps[i].empty() || comps[i] == ".") {
                        comps.erase(comps.begin() + i);
                    } else if (comps[i] == "..") {
                        if (i > 0 && comps[i - 1] != "..") {
                            comps.erase(comps.begin() + i - 1, comps.begin() + i + 1);
                            i -= 1;
                        } else if (i == 0 && pystring::endswith(prefix, "\\")) {
                            comps.erase(comps.begin() + i);
                        } else {
                            i += 1;
                        }
                    } else {
                        i += 1;
                    }
                }
                // If the path is now empty, substitute '.'
                if (prefix.empty() && comps.empty()) {
                    comps.push_back(".");
                }
                return prefix + pystring::join("\\", comps);
            }
            // Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A/B.
            // It should be understood that this may change the meaning of the path
            // if it contains symbolic links!
            // Normalize path, eliminating double slashes, etc.
            
            std::string normpath_posix(std::string const& p) {
                if (p.empty()) { return "."; }
                std::string path = p;
                int initial_slashes = pystring::startswith(path, "/") ? 1 : 0;
                
                // POSIX allows one or two initial slashes, but treats three or more
                // as single slash.
                if (initial_slashes && pystring::startswith(path, "//") &&
                    !pystring::startswith(path, "///")) { initial_slashes = 2; }
                
                std::vector<std::string> comps, new_comps;
                pystring::split(path, comps, "/");
                for (unsigned int i = 0; i < comps.size(); ++i) {
                    std::string comp = comps[i];
                    if (comp.empty() || comp == ".") { continue; }
                    if ((comp != "..") || ((initial_slashes == 0) && new_comps.empty()) ||
                        (!new_comps.empty() && new_comps[new_comps.size() - 1] == "..")) {
                        new_comps.push_back(comp);
                    } else if (!new_comps.empty()) {
                        new_comps.pop_back();
                    }
                }
                path = pystring::join("/", new_comps);
                if (initial_slashes > 0) { path = pystring::mul("/", initial_slashes) + path; }
                if (path.empty()) { return "."; }
                return path;
            }
            
            std::string normpath(std::string const& path) {
#ifdef WINDOWS
                return normpath_nt(path);
#else
                return normpath_posix(path);
#endif
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////
            ///
            ///
            
            // Split the extension from a pathname.
            // Extension is everything from the last dot to the end, ignoring
            // leading dots.  Returns "(root, ext)"; ext may be empty.
            // It is always true that root + ext == p
            
            void splitext_generic(std::string& root, std::string& ext, std::string const& p,
                                  std::string const& sep, std::string const& altsep,
                                  std::string const& extsep) {
                int sepIndex = pystring::rfind(p, sep);
                if (!altsep.empty()) {
                    int altsepIndex = pystring::rfind(p, altsep);
                    sepIndex        = std::max(sepIndex, altsepIndex);
                }
                
                int dotIndex = pystring::rfind(p, extsep);
                if (dotIndex > sepIndex) {
                    // Skip all leading dots
                    int filenameIndex = sepIndex + 1;
                    while (filenameIndex < dotIndex) {
                        if (pystring::slice(p, filenameIndex) != extsep) {
                            root = pystring::slice(p, 0, dotIndex);
                            ext  = pystring::slice(p, dotIndex);
                            return;
                        }
                        filenameIndex += 1;
                    }
                }
                root = p;
                ext  = "";
            }
            
            void splitext_nt(std::string& root, std::string& ext, std::string const& path) {
                return splitext_generic(root, ext, path, "\\", "/", ".");
            }
            
            void splitext_posix(std::string& root, std::string& ext, std::string const& path) {
                return splitext_generic(root, ext, path, "/", "", ".");
            }
            
            void splitext(std::string& root, std::string& ext, std::string const& path) {
#ifdef WINDOWS
                return splitext_nt(root, ext, path);
#else
                return splitext_posix(root, ext, path);
#endif
            }
            
        } // namespace path
    }     // namespace os
    
} // namespace pystring
