/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYINDEX_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYINDEX_HPP_

#include <string>
#include <vector>
#include <Python.h>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/interleaved.hh>

namespace py {
    
    namespace index {
        
        using im::byte;
        using im::Index;
        
        
        template <std::size_t D>
        struct PyIndex : public Index<D> {
            static constexpr std::size_t Dimensions = D;
            
            explicit PyIndex(
                PyObject*   tuple_or_iterable_list_object,
                bool        ignore_inconguent_lengths = false);
            
            PyObject* pytuple();
            PyObject* pystring();
            
        };
        
        // struct XYIndex : public PyIndex<2> {};
        // struct XYCIndex : public PyIndex<3> {};
        
        using XY = PyIndex<2>;
        using XYC = PyIndex<3>;
        
        template <std::size_t D>
        struct PyMeta {
            static constexpr std::size_t Dimensions = D;
            friend struct PyIndex<D>;
                
            using index_t       = Index<D>;
            using idx_t         = typename index_t::idx_t;
            using array_t       = std::array<std::size_t, D>;
            using sequence_t    = std::make_index_sequence<D>;
            
            array_t extents;
            array_t strides;
            index_t max_idx;
            std::size_t elem_size;
            array_t min = { 0 };
            
            Meta(void)
                :extents{ 0 }, strides{ 0 }, max_idx{ 0 }, elem_size(0)
                {}
            
        };
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYINDEX_HPP_