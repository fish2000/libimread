
#ifndef LIBIMREAD_PROCESS_JITRESIZE_HH_
#define LIBIMREAD_PROCESS_JITRESIZE_HH_


#include <Halide.h>

#include <iostream>
#include <limits>
#include <sys/time.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

namespace im {
    
    namespace process {
        
        template <typename T = byte>
        using HalImage = Halide::Image<T>;
        
        inline double now() {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            static bool first_call = true;
            static time_t first_sec = 0;
            if (first_call) {
                first_call = false;
                first_sec = tv.tv_sec;
            }
            assert(tv.tv_sec >= first_sec);
            return (tv.tv_sec - first_sec) + (tv.tv_usec / 1000000.0);
        }
        
        enum InterpolationType { BOX, LINEAR, CUBIC, LANCZOS };
        
        Halide::Expr kernel_box(Halide::Expr x);
        Halide::Expr kernel_linear(Halide::Expr x);
        Halide::Expr kernel_cubic(Halide::Expr x);
        Halide::Expr sinc(Halide::Expr x);
        Halide::Expr kernel_lanczos(Halide::Expr x);
        
        struct Resizer {
            InterpolationType interpolation;
            float scaleFactor;
            int schedule;
            bool compiled;
            Halide::ImageParam input;
            Halide::Func final;
            
            Resizer(float sf = 1.0f, InterpolationType it = LINEAR, int sc = 0)
                :interpolation(it), scaleFactor(sf), schedule(sc)
                ,compiled(false)
                ,final(Halide::Func("final"))
                ,input(Halide::Float(32), 3)
                {}
            
            virtual ~Resizer() {}
            
            void compile();
            HalImage<float> process_impl(HalImage<float> in);
            
            template <typename T, typename U = T> inline
            HalImage<T> process(HalImage<U> in) {
                return Halide::cast<T>(
                    process_impl(
                        Halide::cast<float>(in)));
            }
            
            HalImage<float> operator()(HalImage<float> in) {
                return process_impl(in);
            }
        };
        
        template <> inline
        HalImage<float> Resizer::process(HalImage<float> in) {
            return process_impl(in);
        }
        
        template <typename T = byte, typename U = T>
        HalImage<T> resize(HalImage<U> im,
                        float scaleFactor = 1.0f,
                        InterpolationType interpolationType = CUBIC,
                        int schedule = 0) {
                            Resizer resizer(scaleFactor, interpolationType, schedule);
                            return resizer.process(im);
                        }
        
        // Image<float> resize(Image<float> im, float scaleFactor = 1.0f,
        //                     InterpolationType interpolationType = LINEAR,
        //                     int schedule = 0) {
        //
        //     std::cout << "Finished function setup." << std::endl;
        //     // printf("Loading '%s'\n", infile.c_str());
        //     // Image<float> in_png = load<float>(infile);
        //     // int out_width = in_png.width() * scaleFactor;
        //     // int out_height = in_png.height() * scaleFactor;
        //     // Image<float> out(out_width, out_height, 3);
        //     // input.set(in_png);
        //
        //     int out_width = im.width() * scaleFactor;
        //     int out_height = im.width() * scaleFactor;
        //     Image<float> out(out_width, out_height, 3);
        //     input.set(im);
        //
        //     printf("Resampling from %dx%d to %dx%d using %s interpolation\n",
        //            im.width(), im.height(),
        //            out_width, out_height,
        //            kernelInfo[interpolationType].name);
        //
        //     /// DOUBLE INFINITY!!!!!!!!!
        //     double min = std::numeric_limits<double>::infinity();
        //     double avg = 0.0;
        //     const unsigned int iters = 10;
        //
        //     for (unsigned int i = 0; i < iters; ++i) {
        //         double before = now();
        //         final.realize(out);
        //         double after = now();
        //         double timediff = after - before;
        //
        //         min = (timediff < min) ? timediff : min;
        //         avg = avg + (timediff - avg) / (i + 1);
        //         std::cout << "   " << timediff * 1000 << " (avg=" << avg * 1000 << ")\n";
        //     }
        //     std::cout << " took min=" << min * 1000 << ", avg=" << avg * 1000 << " msec." << std::endl;
        //
        //     return out;
        // }
        
    }
    
}


#endif /// LIBIMREAD_PROCESS_JITRESIZE_HH_