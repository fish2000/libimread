
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BASE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BASE_HH_

#include <type_traits>
#include <Python.h>
#include <structmember.h>
#include "../detail.hpp"

namespace py {
    
    namespace ext {
        
        struct Model;
        
        namespace detail {
            
            template <typename T>
            using void_t = std::conditional_t<true, void, T>;
            
            template <bool B, typename T = void>
            using disable_if = std::enable_if<!B, T>;
            
            template <bool B, typename T = void>
            using disable_if_t = std::enable_if_t<!B, T>;
            
            template <typename T>
            using is_python_model_t = std::is_base_of<py::ext::Model, T>::type;
            
            template <class, class = void>
            struct has_weakrefs : std::false_type {};
            template <class T>
            struct has_weakrefs<T, void_t<typename T::weakrefs>> : is_python_model_t<T> {};
            
            template <typename T>
            bool is_python_model_v = std::is_base_of<py::ext::Model, T>::value;
            
            template <class X, class Y = void>
            bool has_weakrefs_v = has_weakrefs<X, Y>::value;
            template <class X>
            bool has_weakrefs_v = has_weakrefs<X>::value;
            
        }
        
        struct Model {
            
            static PyObject* new(PyTypeObject* type, std::size_t newsize = 0) {
                return type->tp_alloc(type, newsize);
            }
            
            static void delete(PyTypeObject* type, PyObject* pyobject) {
                type->tp_free(pyobject);
            }
            
            static PyObject* gc_new(PyTypeObject* type, std::size_t newsize = 0) {
                PyObject* out = _PyObject_GC_New(type);
                PyObject_GC_Track(out);
                return out;
            }
            
            static void gc_delete(PyTypeObject* type, PyObject* pyobject) {
                PyObject_GC_UnTrack(reinterpret_cast<void*>(pyobject));
                Model::delete(type, pyobject);
            }
            
            static void clear_weak_refs(PyObject* pyobject) {
                if (pyobject->weakrefs != nullptr) {
                    PyObject_ClearWeakRefs(pyobject);
                }
            }
            
        };
        
        template <typename ModelType,
                  bool CollectGarbage = false>
        struct ModelBase : public Model {
            
            static ModelType* typed(PyObject* pyobject) {
                return reinterpret_cast<ModelType*>(pyobject);
            }
            
            static ModelType* typed(void* voidpointer) {
                return reinterpret_cast<ModelType*>(voidpointer);
            }
            
            static bool check(PyObject* pyobject) {
                return Py_TYPE(pyobject) == ModelType::type_ptr();
            }
            
            static bool check(void* voidpointer) {
                return Py_TYPE(voidpointer) == ModelType::type_ptr();
            }
            
            static bool check(ModelType* object) {
                return true;
            }
            
            void* operator new(std::size_t newsize) {
                if (CollectGarbage) {
                    return reinterpret_cast<void*>(Model::gc_new(
                                                   ModelType::type_ptr()));
                } else {
                    return reinterpret_cast<void*>(Model::new(
                                                   ModelType::type_ptr()));
                }
            }
            
            template <typename detail::disable_if_t<
                               detail::has_weakrefs_v<ModelType>,
                      int> = 0>
            static void delete_impl(ModelType* object) {}
            
            template <typename std::enable_if_t<
                               detail::has_weakrefs_v<ModelType>,
                      int> = 0>
            static void delete_impl(ModelType* object) {
                Model::clear_weak_refs(py::convert(object));
            }
            
            void operator delete(void* voidpointer) {
                ModelType* object = ModelType::typed(voidpointer);
                ModelType::delete_impl(object);
                object->cleanup();
                if (CollectGarbage) {
                    Model::delete(ModelType::type_ptr(),
                                  py::convert(object));
                } else {
                    Model::gc_delete(ModelType::type_ptr(),
                                     py::convert(object));
                }
            }
        };
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BASE_HH_