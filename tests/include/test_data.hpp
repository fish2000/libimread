
#ifndef IMREAD_TESTDATA_HPP_
#define IMREAD_TESTDATA_HPP_

#include <string>

namespace im {
    
    static const std::string basedir = "/Users/fish/Dropbox/libimread/tests/data";
    
    static const int num_jpg = 3;
    static const std::string jpg[] = {
        "10954288_342637995941364_1354507656_n.jpg",
    "IMG_4332.jpg",
    "tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"
    };
    
    static const int num_jpeg = 1;
    static const std::string jpeg[] = {
        "IMG_7333.jpeg"
    };
    
    static const int num_png = 4;
    static const std::string png[] = {
        "grad_32_rrt_srgb.png",
    "marci_512_srgb.png",
    "marci_512_srgb8.png",
    "roses_512_rrt_srgb.png"
    };
    
    static const int num_tif = 1;
    static const std::string tif[] = {
        "ptlobos.tif"
    };
    
    static const int num_tiff = 0;
    static const std::string tiff[] = {
    
    };
}

#endif /// IMREAD_TESTDATA_HPP_

