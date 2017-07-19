/*
=========================================================================

  Program:   BioImageXD
  Module:    $RCSfile: LSMFormat.h,v $
  Language:  C++
  Date:      $Date: 2003/08/22 14:46:02 $
  Version:   $Revision: 1.39 $

 This is an open-source copyright as follows:
 Copyright (c) 2004-2008 BioImageXD Development Team

 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Modified source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
 IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.


=========================================================================
*/
#ifndef LIBIMREAD_IO_LSM_HH_
#define LIBIMREAD_IO_LSM_HH_

// .NAME LSMFormat - read LSM files
// .SECTION Description
// LSMFormat is a source object that reads LSM files.
// It should be able to read most any LSM file
//
// .SECTION Thanks
// This class was developed as a part of the BioImageXD Project.
// The BioImageXD project includes the following people:
//
// Dan White <dan@chalkie.org.uk>
// Kalle Pahajoki <kalpaha@st.jyu.fi>
// Pasi Kankaanpää <ppkank@bytl.jyu.fi>
//

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class LSMFormat : public ImageFormatBase<LSMFormat> {
        public:
            using can_read = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x49\x49\x2a\x00", 4)
                },
                _suffixes = { "lsm", "tif", "tiff" },
                _mimetype = "image/lsm"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
    };
    
    namespace format {
        using LSM = LSMFormat;
    }
    
}

#endif /// LIBIMREAD_IO_LSM_HH_
