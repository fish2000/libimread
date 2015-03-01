// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_FORMATS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_FORMATS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <memory>

namespace im {

    std::unique_ptr<ImageFormat> get_format(const char*);
    const char* magic_format(byte_source*);

}

#endif // LPC_FORMATS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012