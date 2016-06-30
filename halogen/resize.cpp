
#include <array>
#include <algorithm>
#include <type_traits>

#include "Halide.h"

using namespace Halide;

namespace {
    
    enum Interpolation {
        BOX,
        LINEAR,
        CUBIC,
        LANCZOS
    };
    
    /// Generator definition
    class Resize : public Generator<Resize> {
        
        public:
            
            static Expr box(Expr x) {
                Expr xx = abs(x);
                return select(xx <= 0.5f,
                    1.0f,
                    0.0f);
            }
            
            static Expr linear(Expr x) {
                Expr xx = abs(x);
                return select(xx < 1.0f,
                    1.0f - xx,
                    0.0f);
            }
            
            static Expr cubic(Expr x) {
                Expr xx = abs(x);
                Expr xx2 = xx * xx;
                Expr xx3 = xx2 * xx;
                float a = -0.5f;
                
                return select(xx < 1.0f,
                    (a + 2.0f) * xx3 - (a + 3.0f) * xx2 + 1,
                    select(xx < 2.0f,
                        a * xx3 - 5 * a * xx2 + 8 * a * xx - 4.0f * a,
                        0.0f)
                    );
            }
            
            static Expr sinc(Expr x) {
                return sin(float(M_PI) * x) / x;
            }
            
            static Expr lanczos(Expr x) {
                Expr value = Resize::sinc(x) * Resize::sinc(x/3);
                
                // Take care of singularity at zero
                value = select(x == 0.0f,
                    1.0f,
                    value);
                
                // Clamp to zero out of bounds
                value = select(x > 3 || x < -3,
                    0.0f,
                    value);
                
                return value;
            }
            
            using kernel_f = typename std::add_pointer<Expr(Expr)>::type;
            
            struct KernelInfo {
                char const* name;
                float size;
                kernel_f kernel; /// Halide::Expr (*kernel)(Halide::Expr);
            };
            
            const std::array<KernelInfo, 4> kernels{{
                { "box",        0.5f,   Resize::box      },
                { "linear",     1.0f,   Resize::linear   },
                { "cubic",      2.0f,   Resize::cubic    },
                { "lanczos",    3.0f,   Resize::lanczos  }
            }};
            
            /// COMPILE-TIME GENERATOR PARAMS
            GeneratorParam<Halide::Type>     input_type{  "input_type",   UInt(8) };
            GeneratorParam<Halide::Type>    output_type{ "output_type",   UInt(8) };
            GeneratorParam<Interpolation> interpolation{ "interpolation", Interpolation::CUBIC,
                                                                {{ "box", Interpolation::BOX },
                                                              { "linear", Interpolation::LINEAR },
                                                               { "cubic", Interpolation::CUBIC },
                                                             { "lanczos", Interpolation::LANCZOS } }};
            
            /// RUNTIME PARAMS
            ImageParam input{ input_type, 3, "input"              };
            Param<float> scale_factor{       "scale_factor", 1.0f };
            
            // Var x("x"), y("y"), c("c"), k("k");
            Var x, y, c, k;
            
            Func build() {
                using kernel_t = Resize::KernelInfo;
                
                /// Reset `input` ImageParam to reflect `input_type`
                input = ImageParam{ input_type, 3, "input" };
                // float scaleFactor = (float)scale_factor;
                
                /// Set up repeating boundary edge conditions on input image
                Func clamped = BoundaryConditions::repeat_edge(input);
                
                /// For downscaling, widen the interpolation kernel to perform lowpass filtering.
                Expr kernelScaling = min(scale_factor, 1.0f);
                Expr kernelSize = Resize::kernels[interpolation].size / kernelScaling;
                
                /// source[xy] are the (non-integer) coordinates inside the source image
                Expr sourcex = (x + 0.5f) / scale_factor;
                Expr sourcey = (y + 0.5f) / scale_factor;
                
                /// Initialize interpolation kernels. Since we allow an arbitrary
                /// scaling factor, the filter coefficients are different for each x
                /// and y coordinate.
                Func kernelx("kernelx");
                Func kernely("kernely");
                Expr beginx = Halide::cast<int>(sourcex - kernelSize + 0.5f);
                Expr beginy = Halide::cast<int>(sourcey - kernelSize + 0.5f);
                RDom domx(0, Halide::cast<int>(2.0f * kernelSize) + 1, "domx");
                RDom domy(0, Halide::cast<int>(2.0f * kernelSize) + 1, "domy");
                
                {
                    kernel_t const& info = Resize::kernels[interpolation];
                    Func kx, ky;
                    kx(x, k) = info.kernel((k + beginx - sourcex) * kernelScaling);
                    ky(y, k) = info.kernel((k + beginy - sourcey) * kernelScaling);
                    kernelx(x, k) = kx(x, k) / sum(kx(x, domx));
                    kernely(y, k) = ky(y, k) / sum(ky(y, domy));
                }
                
                /// Perform separable resizing
                Func resized_x("resized_x");
                Func resized_y("resized_y");
                Func penultimate("penultimate");
                Func final("final");
                
                resized_x(x, y, c) = sum(kernelx(x, domx) * Halide::cast<float>(clamped(domx + beginx, y, c)));
                resized_y(x, y, c) = sum(kernely(y, domy) * resized_x(x, domy + beginy, c));
                penultimate(x, y, c) = clamp(resized_y(x, y, c), 0.0f, 1.0f);
                final(x, y, c) = cast(output_type, penultimate(x, y, c));
                
                /// Scheduling
                // bool parallelize = (schedule >= 2);
                // bool vectorize = (schedule == 1 || schedule == 3);
                bool parallelize = true,
                     vectorize = true;
                kernelx.compute_root();
                kernely.compute_at(final, y);
                if (vectorize) {
                    resized_x.vectorize(x, natural_vector_size(output_type)); /// originally 4
                    final.vectorize(x, natural_vector_size(output_type)); /// originally 4
                }
                if (parallelize) {
                    Var yo, yi;
                    final.split(y, yo, y, 32).parallel(yo);
                    resized_x.store_at(final, yo).compute_at(final, y);
                } else {
                    resized_x.store_at(final, c).compute_at(final, y);
                }
                
                return final;
            }
            
    };
    
    /// Generator registration
    Halide::RegisterGenerator<Resize> register_resize{ "resize" };
    
} /* namespace (anon.) */