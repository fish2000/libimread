// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// EtcCompressor is a derived Compressor class that implements ETC
// (Ericsson Texture Compression) image compression and decompression.
// It supports ETC1 compression for RGB images. It does not support
// RGBA images at present.
//
// Compression should work correctly on source images of any size.
// Since ETC compression operates on 4x4 blocks of pixels, the
// compressed output image always has dimensions that are multiples of
// 4.  When compressing, if either of the original image dimensions is
// not a multiple of 4, the last row or column of pixels in the
// original image is replicated to simulate the correct size in the
// compressed output.
//
// The compressed_height and compressed_width members of the
// CompressionSpec can be used to specify a different size for the
// resulting compressed image. For example, they can be used to
// compress to an image whose dimensions are (larger) powers of 2.
// Results in such cases are similar to those of the Pad() function.
//
// Decompression has similar sizing rules, allowing a subimage of a
// compressed image to be decompressed into an image of the
// appropriate size.
//
// Reduction of a compressed image requires both image dimensions be a
// multiple of 8.

#ifndef IMAGE_COMPRESSION_PUBLIC_ETC_COMPRESSOR_H_
#define IMAGE_COMPRESSION_PUBLIC_ETC_COMPRESSOR_H_

#include <cstddef>
#include <vector>

#include "image_compression/public/compressed_image.h"
#include "image_compression/public/compressor.h"

namespace imagecompression {
    
    class EtcCompressor : public Compressor {
        
        public:
            // This enum is used to indicate what strategy to use when compressing
            // 4x4 blocks.
            enum CompressionStrategy {
                kSplitHorizontally, // Always split into 2 rows x 4 columns.
                kSplitVertically,   // Always split into 4 rows x 2 columns.
                kSmallerError,      // Choose the split that produces the smaller error.
                kHeuristic,         // Use a faster heuristic instead of exhaustive search.
            };
            
            EtcCompressor();
            virtual ~EtcCompressor();
            
            // Sets/returns the compression strategy. This specifies what to do when
            // compressing each 4x4 block of pixels. ETC operates on a 2x4 or 4x2
            // subblock at a time, so it can split a 4x4 block either vertically or
            // horizontally. The default is kSmallerError.
            void SetCompressionStrategy(CompressionStrategy strategy) { compression_strategy_ = strategy; }
            CompressionStrategy GetCompressionStrategy() const { return compression_strategy_; }
            
            virtual bool SupportsFormat(CompressedImage::Format format) const;
            
            virtual bool IsValidCompressedImage(CompressedImage const& image);
            
            virtual size_t ComputeCompressedDataSize(CompressedImage::Format format, uint32_t height,
                                                     uint32_t width);
            
            virtual bool Compress(CompressedImage::Format format,
                                  uint32_t height, uint32_t width,
                                  uint32_t padding_bytes_per_row,
                                  const uint8_t* buffer,
                                  CompressedImage* image);
            
            virtual bool Decompress(CompressedImage const& image,
                                    std::vector<uint8_t>* decompressed_buffer);
            
            virtual bool Downsample(CompressedImage const& image,
                                    CompressedImage* downsampled_image);
            
            virtual bool Pad(CompressedImage const& image,
                             uint32_t padded_height, uint32_t padded_width,
                             CompressedImage* padded_image);
            
            virtual bool CompressAndPad(CompressedImage::Format format,
                                        uint32_t height, uint32_t width,
                                        uint32_t padded_height, uint32_t padded_width,
                                        uint32_t padding_bytes_per_row,
                                        const uint8_t* buffer,
                                        CompressedImage* padded_image);
            
            virtual bool CreateSolidImage(CompressedImage::Format format,
                                          uint32_t height, uint32_t width,
                                          const uint8_t* color,
                                          CompressedImage* image);
            
            virtual bool CopySubimage(CompressedImage const& image,
                                      uint32_t start_row, uint32_t start_column,
                                      uint32_t height, uint32_t width,
                                      CompressedImage* subimage);
        
        private:
            CompressionStrategy compression_strategy_;
    };

} // namespace imagecompression

#endif // IMAGE_COMPRESSION_PUBLIC_ETC_COMPRESSOR_H_
