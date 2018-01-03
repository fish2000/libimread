/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ACCESSORS_HH_
#define LIBIMREAD_ACCESSORS_HH_

#include <vector>
#include <libimread/ext/arrayview.hh>


#define IMAGE_ACCESSOR_ROWP_AS(__pointer__)                                                         \
                                                                                                    \
    template <typename T> inline                                                                    \
    T* rowp_as(const int r) const {                                                                 \
        return static_cast<T*>(__pointer__->rowp(r));                                               \
    }

#define IMAGE_ACCESSOR_VIEW(__pointer__)                                                            \
                                                                                                    \
    template <typename T = value_type> inline                                                       \
    av::strided_array_view<T, 3> view(int X = -1,                                                   \
                                      int Y = -1,                                                   \
                                      int Z = -1) const {                                           \
        if (X == -1) { X = __pointer__->dim(0); }                                                   \
        if (Y == -1) { Y = __pointer__->dim(1); }                                                   \
        if (Z == -1) { Z = __pointer__->dim(2); }                                                   \
        return av::strided_array_view<T, 3>(static_cast<T*>(__pointer__->rowp(0)),                  \
                                            { X, Y, Z },                                            \
                                            { __pointer__->stride(0),                               \
                                              __pointer__->stride(1),                               \
                                              __pointer__->stride(2) });                            \
    }

#define IMAGE_ACCESSOR_ALLROWS(__pointer__)                                                         \
                                                                                                    \
    template <typename T> inline                                                                    \
    std::vector<T*> allrows() const {                                                               \
        using pointervec_t = std::vector<T*>;                                                       \
        pointervec_t rows;                                                                          \
        const int h = __pointer__->dim(0);                                                          \
        for (int r = 0; r != h; ++r) {                                                              \
            rows.push_back(static_cast<T*>(__pointer__->rowp(r)));                                  \
        }                                                                                           \
        return rows;                                                                                \
    }

/// TODO: Do not fill plane vector?
/// TODO: arrayview transpose?

#define IMAGE_ACCESSOR_PLANE(__pointer__)                                                           \
                                                                                                    \
    template <typename T, typename U = value_type> inline                                           \
    std::vector<T> plane(int idx) const {                                                           \
        /* types */                                                                                 \
        using planevec_t = std::vector<T>;                                                          \
        using view_t = av::strided_array_view<T, 3>;                                                \
        if (idx >= __pointer__->planes()) { return planevec_t{}; }                                  \
        /* image dimensions */                                                                      \
        const int w = __pointer__->dim(0);                                                          \
        const int h = __pointer__->dim(1);                                                          \
        const int siz = w * h;                                                                      \
        view_t viewer = __pointer__->template view<T>();                                            \
        /* fill plane vector */                                                                     \
        planevec_t out;                                                                             \
        out.resize(siz, 0);                                                                         \
        for (int x = 0; x < w; ++x) {                                                               \
            for (int y = 0; y < h; ++y) {                                                           \
                out[y * w + x] = static_cast<U>(viewer[{x, y, idx}]);                               \
            }                                                                                       \
        }                                                                                           \
        return out;                                                                                 \
    }

#define IMAGE_ACCESSOR_ALLPLANES(__pointer__)                                                       \
                                                                                                    \
    template <typename T, typename U = value_type> inline                                           \
    std::vector<std::vector<T>> allplanes(int lastplane = 255) const {                              \
        using planevec_t = std::vector<T>;                                                          \
        using pixvec_t = std::vector<planevec_t>;                                                   \
        const int planecount = std::min(__pointer__->planes(), lastplane);                          \
        pixvec_t out;                                                                               \
        out.reserve(planecount);                                                                    \
        for (int idx = 0; idx < planecount; ++idx) {                                                \
            out.emplace_back(__pointer__->template plane<T, U>(idx));                               \
        }                                                                                           \
        return out;                                                                                 \
    }

#endif /// LIBIMREAD_ACCESSORS_HH_