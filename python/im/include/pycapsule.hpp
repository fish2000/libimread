/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYCAPSULE_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYCAPSULE_HPP_

#include <type_traits>
#include <cstdlib>
#include <Python.h>

namespace py {
    
    namespace detail {
        
        template <typename T>
        bool null_or_void = std::is_void<T>::value ||
                            std::is_null_pointer<T>::value;
        
    };
    
    namespace capsule {
        
        using destructor_t = PyCapsule_Destructor;
        
        /// Default capsule destructor template
        template <typename Pointer, typename Context>
        destructor_t decapsulator = [](PyObject* capsule) {
            /// PyCapsule_Get* calls are guaranteed for valid capsules
            if (PyCapsule_IsValid(capsule, PyCapsule_GetName(capsule))) {
                char const* name = PyCapsule_GetName(capsule);
                Context* context = (Context*)PyCapsule_GetContext(capsule);
                if (context) { delete context;      context = nullptr; }
                Pointer* pointer = (Pointer*)PyCapsule_GetPointer(capsule, name);
                if (pointer) { delete pointer;      pointer = nullptr; }
                if (name) { std::free((void*)name); name = nullptr;    }
            }
        };
        
        template <typename Pointer, typename Context>
        PyObject* encapsulate(Pointer* pointer,
                              Context* context = nullptr,
                              char const* name = nullptr,
                              destructor_t destructor = decapsulator<Pointer, Context>) {
            if (!pointer) {
                PyErr_SetString(PyExc_ValueError,
                    "Can't encapsulate(nullptr)");
                return nullptr;
            }
            PyObject* capsule = PyCapsule_New((void*)pointer,
                                              name ? ::strdup(name) : name,
                                              destructor);
            if (!capsule) {
                PyErr_SetString(PyExc_ValueError,
                    "Failure creating PyCapsule");
            } else if (PyCapsule_SetContext(capsule, (void*)context) != 0) {
                /// context can be nullptr w/o invalidating capsule
                PyErr_SetString(PyExc_ValueError,
                    "Failure assigning PyCapsule context");
            } 
            return capsule;
        }
        
    }; /* namespace capsule */
    
    namespace cob {
        
        using single_destructor_t = std::add_pointer_t<void(void*)>;
        using double_destructor_t = std::add_pointer_t<void(void*, void*)>;
        
        template <typename Pointer, typename>
        single_destructor_t default_single = [](void* voidptr) {
            Pointer* pointer = (Pointer*)voidptr;
            if (pointer) { delete pointer; pointer = nullptr; }
        };
        
        template <typename Pointer, typename Context>
        double_destructor_t default_double = [](void* voidptr, void* voidctx) {
            Context* context = (Context*)voidctx;
            if (context) { delete context; context = nullptr; }
            Pointer* pointer = (Pointer*)voidptr;
            if (pointer) { delete pointer; pointer = nullptr; }
        };
        
        template <typename Pointer,
                  typename Destructor = single_destructor_t>
        PyObject* objectify(Pointer* pointer,
                            Destructor destructor = default_single<Pointer, std::nullptr_t>) {
            PyObject* cob = nullptr;
            if (!pointer) {
                PyErr_SetString(PyExc_ValueError,
                    "Can't objectify(nullptr)");
            } else {
                cob = PyCObject_FromVoidPtr((void*)pointer, destructor);
            }
            if (!cob) {
                PyErr_SetString(PyExc_ValueError,
                    "Failure creating PyCObject");
            }
            return cob;
        }
        
        template <typename Pointer,
                  typename Context,
                  typename Destructor = double_destructor_t>
        PyObject* objectify(Pointer* pointer,
                            Context* context,
                            Destructor destructor = default_double<Pointer, Context>) {
            PyObject* cob = nullptr;
            if (!pointer) {
                PyErr_SetString(PyExc_ValueError,
                    "Can't objectify(nullptr, ...)");
            } else {
                cob = PyCObject_FromVoidPtrAndDesc((void*)pointer, (void*)context, destructor);
            }
            if (!cob) {
                PyErr_SetString(PyExc_ValueError,
                    "Failure creating PyCObject with context");
            }
            return cob;
        }
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYCAPSULE_HPP_