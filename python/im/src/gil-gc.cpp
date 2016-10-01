
#include "gil-gc.hpp"

namespace py {
    
    namespace gil {
        
        #define PENULTIMATE(object) _Py_AS_GC(object)->gc.gc_prev
        
        ensure_t::ensure_t() {
            int activations = ++ensure_t::activations;
            if (activations >= 1 && !ensure_t::state) {
                ensure_t::state = ::PyGILState_Ensure();
            }
            activationID = activations;
        }
        
        ensure_t::~ensure_t() {
            if (--ensure_t::activations <= 0) {
                ::PyGILState_Release(ensure_t::state);
                ensure_t::activations = 0;
            }
        }
        
        gilstate_t ensure_t::state;
        std::atomic<int> ensure_t::activations{ 0 };
        std::atomic<int> basura_t::nesting{ 0 };
        std::recursive_mutex basura_t::mute;
        std::mutex basura_t::latermute;
        object_t* basura_t::delete_later = nullptr;
        
        basura_t::basura_t(object_t* object)
            :lock{ basura_t::mute }
            ,state(), container(object)
            {
                untrack();
                basura_t::nest();
            }
        
        basura_t::basura_t(void* voidptr)
            :lock{ basura_t::mute }
            ,state(), container(reinterpret_cast<object_t*>(voidptr))
            {
                untrack();
                basura_t::nest();
            }
        
        basura_t::~basura_t() {
            if (basura_t::is_nested_deep()) {
                deposit();
            } else {
                basura_t::unnest();
                basura_t::destroy_chain();
            }
        }
        
        void basura_t::deposit(object_t* object) {
            // _PyTrash_deposit_object(object);
            laterlock_t ulock(basura_t::latermute);
            PENULTIMATE(object) = (gc_head_t*)basura_t::delete_later;
            basura_t::delete_later = object;
        }
        
        void basura_t::deposit(void* voidptr) {
            deposit(reinterpret_cast<object_t*>(voidptr));
        }
        
        void basura_t::deposit() {
            basura_t::deposit(container);
        }
        
        void basura_t::untrack() {
            ::PyObject_GC_UnTrack(container);
        }
        
        void basura_t::nest()      { ++basura_t::nesting; }
        void basura_t::unnest()    { --basura_t::nesting; }
        
        bool basura_t::is_nested_shallow() {
            // _PyTrash_delete_nesting = basura_t::nesting;
            // return _PyTrash_delete_nesting < basura_t::unwind_max;
            return basura_t::nesting < basura_t::unwind_max;
        }
        
        bool basura_t::is_nested_deep() {
            // _PyTrash_delete_nesting = basura_t::nesting;
            // return _PyTrash_delete_nesting > basura_t::unwind_max;
            return basura_t::nesting > basura_t::unwind_max;
        }
        
        void basura_t::destroy_chain() {
            // if (basura_t::delete_later && basura_t::nesting <= 0) {
                // std::lock_guard<std::mutex> lock(basura_t::mute);
                // _PyTrash_delete_nesting = basura_t::nesting;
                // _PyTrash_delete_later = basura_t::delete_later;
                // ::_PyTrash_destroy_chain();
                // basura_t::nesting = _PyTrash_delete_nesting;
                // basura_t::delete_later = _PyTrash_delete_later;
            // }
            
            object_t* object;
            destructor dealloc;
            
            while (basura_t::delete_later) {
                laterlock_t ulock(basura_t::latermute);
                
                object = basura_t::delete_later;
                dealloc = object->ob_type->tp_dealloc;
                basura_t::delete_later = (object_t*)PENULTIMATE(object);
                
                // assert(object->ob_refcnt == 0);
                nest();
                (*dealloc)(object);
                unnest();
            }
            
        }
        
    } /* namespace gil */
    
} /* namespace py */
