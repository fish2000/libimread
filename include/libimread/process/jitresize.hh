
#ifndef LIBIMREAD_PROCESS_JITRESIZE_HH_
#define LIBIMREAD_PROCESS_JITRESIZE_HH_


#include <Halide.h>
using namespace Halide;

#include <iostream>
#include <limits>
#include <sys/time.h>

namespace im {
    
    namespace process {
        
        double now() {
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
        
        Expr kernel_box(Expr x);
        Expr kernel_linear(Expr x);
        Expr kernel_cubic(Expr x);
        Expr sinc(Expr x);
        Expr kernel_lanczos(Expr x);
        
        struct Resizer {
            float scaleFactor;
            InterpolationType interpolation;
            int schedule;
            bool compiled;
            ImageParam input;
            Func final;
            
            Resizer(float sf = 1.0f, InterpolationType it = LINEAR, int sc = 0)
                :scaleFactor(sf), interpolation(it), schedule(sc)
                ,compiled(false)
                ,final(Func("final"))
                ,input(Float(32), 3)
                {}
            
            virtual ~Resizer() {}
            
            void compile();
            Image<float> process_impl(Image<float> in);
            
            template <typename T, typename U = T>
            Image<T> process(Image<U> in) {
                return cast<T>(process_impl(cast<float>(in)));
            }
            
            Image<float> operator()(Image<float> in) {
                return process_impl(in);
            }
        };
        
        template <>
        Image<float> Resizer::process(Image<float> in) {
            return process_impl(in);
        }
        
        /*
        Image<float> resize(Image<float> im, float scaleFactor = 1.0f,
                            InterpolationType interpolationType = LINEAR,
                            int schedule = 0) {
            
            std::cout << "Finished function setup." << std::endl;
            // printf("Loading '%s'\n", infile.c_str());
            // Image<float> in_png = load<float>(infile);
            // int out_width = in_png.width() * scaleFactor;
            // int out_height = in_png.height() * scaleFactor;
            // Image<float> out(out_width, out_height, 3);
            // input.set(in_png);
            
            int out_width = im.width() * scaleFactor;
            int out_height = im.width() * scaleFactor;
            Image<float> out(out_width, out_height, 3);
            input.set(im);
            
            printf("Resampling from %dx%d to %dx%d using %s interpolation\n",
                   im.width(), im.height(),
                   out_width, out_height,
                   kernelInfo[interpolationType].name);
            
            /// DOUBLE INFINITY!!!!!!!!!
            double min = std::numeric_limits<double>::infinity();
            double avg = 0.0;
            const unsigned int iters = 10;
            
            for (unsigned int i = 0; i < iters; ++i) {
                double before = now();
                final.realize(out);
                double after = now();
                double timediff = after - before;
                
                min = (timediff < min) ? timediff : min;
                avg = avg + (timediff - avg) / (i + 1);
                std::cout << "   " << timediff * 1000 << " (avg=" << avg * 1000 << ")\n";
            }
            std::cout << " took min=" << min * 1000 << ", avg=" << avg * 1000 << " msec." << std::endl;
        
            return out;
        }
        */
        
        
    }
    
}


#endif /// LIBIMREAD_PROCESS_JITRESIZE_HH_