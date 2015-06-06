/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <Halide.h>
using namespace Halide;

#include <libimread/process/jitresize.hh>

namespace im {

    namespace process {
        
        Expr kernel_box(Expr x) {
            Expr xx = abs(x);
            return select(xx <= 0.5f, 1.0f, 0.0f);
        }
        
        Expr kernel_linear(Expr x) {
            Expr xx = abs(x);
            return select(xx < 1.0f, 1.0f - xx, 0.0f);
        }
        
        Expr kernel_cubic(Expr x) {
            Expr xx = abs(x);
            Expr xx2 = xx * xx;
            Expr xx3 = xx2 * xx;
            float a = -0.5f;
            
            return select(xx < 1.0f, (a + 2.0f) * xx3 - (a + 3.0f) * xx2 + 1,
                          select (xx < 2.0f, a * xx3 - 5 * a * xx2 + 8 * a * xx - 4.0f * a,
                                  0.0f));
        }
        
        Expr sinc(Expr x) {
            return sin(float(M_PI) * x) / x;
        }
        
        Expr kernel_lanczos(Expr x) {
            Expr value = sinc(x) * sinc(x/3);
            value = select(x == 0.0f, 1.0f, value); // Take care of singularity at zero
            value = select(x > 3 || x < -3, 0.0f, value); // Clamp to zero out of bounds
            return value;
        }
        
        struct KernelInfo {
            const char *name;
            float size;
            Expr (*kernel)(Expr);
        };
        
        static KernelInfo kernelInfo[] = {
            { "box",        0.5f,   kernel_box      },
            { "linear",     1.0f,   kernel_linear   },
            { "cubic",      2.0f,   kernel_cubic    },
            { "lanczos",    3.0f,   kernel_lanczos  }
        };
        
        void Resizer::compile() {
            
            /// Start with a param to configure the processing pipeline
            //ImageParam input(Float(32), 3);
            Var x("x"), y("y"), c("c"), k("k");
            Func clamped = BoundaryConditions::repeat_edge(input);
            
            /// For downscaling, widen the interpolation kernel to perform lowpass
            /// filtering.
            float kernelScaling = std::min(scaleFactor, 1.0f);
            float kernelSize = kernelInfo[interpolation].size / kernelScaling;
            
            /// source[xy] are the (non-integer) coordinates inside the source image
            Expr sourcex = (x + 0.5f) / scaleFactor;
            Expr sourcey = (y + 0.5f) / scaleFactor;
            
            /// Initialize interpolation kernels. Since we allow an arbitrary
            /// scaling factor, the filter coefficients are different for each x
            /// and y coordinate.
            Func kernelx("kernelx"), kernely("kernely");
            Expr beginx = cast<int>(sourcex - kernelSize + 0.5f);
            Expr beginy = cast<int>(sourcey - kernelSize + 0.5f);
            RDom domx(0, static_cast<int>(2.0f * kernelSize) + 1, "domx");
            RDom domy(0, static_cast<int>(2.0f * kernelSize) + 1, "domy");
            
            {
                const KernelInfo &info = kernelInfo[interpolation];
                Func kx, ky;
                kx(x, k) = info.kernel((k + beginx - sourcex) * kernelScaling);
                ky(y, k) = info.kernel((k + beginy - sourcey) * kernelScaling);
                kernelx(x, k) = kx(x, k) / sum(kx(x, domx));
                kernely(y, k) = ky(y, k) / sum(ky(y, domy));
            }
            
            /// Perform separable resizing
            Func resized_x("resized_x");
            Func resized_y("resized_y");
            resized_x(x, y, c) = sum(kernelx(x, domx) * cast<float>(clamped(domx + beginx, y, c)));
            resized_y(x, y, c) = sum(kernely(y, domy) * resized_x(x, domy + beginy, c));
            
            //Func final("final");
            final(x, y, c) = clamp(resized_y(x, y, c), 0.0f, 1.0f);
            
            /// Scheduling
            bool parallelize = (schedule >= 2);
            bool vectorize = (schedule == 1 || schedule == 3);
            
            kernelx.compute_root();
            kernely.compute_at(final, y);
            
            if (vectorize) {
                resized_x.vectorize(x, 4);
                final.vectorize(x, 4);
            }
            
            if (parallelize) {
                Var yo, yi;
                final.split(y, yo, y, 32).parallel(yo);
                resized_x.store_at(final, yo).compute_at(final, y);
            } else {
                resized_x.store_at(final, c).compute_at(final, y);
            }
            
            Target target = get_jit_target_from_environment();
            final.compile_jit(target);
            compiled = true;
        }
        
        Image<float> Resizer::process_impl(Image<float> in) {
            int out_width = in.width() * scaleFactor;
            int out_height = in.width() * scaleFactor;
            Image<float> out(out_width, out_height, 3);
            input.set(in);
            
            printf("Resampling from %dx%d to %dx%d using %s interpolation\n",
                   in.width(), in.height(),
                   out_width, out_height,
                   kernelInfo[interpolation].name);
            
            final.realize(out);
            return out;
        }
        
        
    }
}