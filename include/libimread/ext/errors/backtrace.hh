/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from http://stackoverflow.com/a/31633962/298171

#pragma once
#include <ostream>

namespace terminator {
    
    void backtrace(std::ostream& _out) noexcept;
    
} /* namespace terminator */

