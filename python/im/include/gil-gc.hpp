
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_GIL_GC_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_GIL_GC_HPP_

#include <atomic>
#include <mutex>

#include <Python.h>

namespace py {
    
    namespace gil {
        
        using gilstate_t = ::PyGILState_STATE;
        using gc_head_t = ::PyGC_Head;
        using object_t = ::PyObject;
        
        struct ensure_t {
            
            mutable int                     activationID = 0;
            
            ensure_t();
            virtual ~ensure_t();
            
            private:
                static gilstate_t           state;
                static std::atomic<int>     activations;
                ensure_t(ensure_t const&);
                ensure_t(ensure_t&&);
                ensure_t& operator=(ensure_t const&);
                ensure_t& operator=(ensure_t&&);
            
        };
        
        using trashlock_t = std::lock_guard<std::recursive_mutex>;
        using laterlock_t = std::lock_guard<std::mutex>;
        
        struct basura_t {
            
            trashlock_t lock;
            ensure_t    state;
            object_t*   container = nullptr;
            
            private:
                
                static constexpr int unwind_max = PyTrash_UNWIND_LEVEL;
                static std::atomic<int> nesting;
                static std::recursive_mutex mute;
                static std::mutex latermute;
                static object_t* delete_later;
            
            public:
                
                explicit basura_t(object_t*);
                explicit basura_t(void*);
                virtual ~basura_t();
                
                static void deposit(object_t* object);
                static void deposit(void* voidptr);
                void deposit();
                void untrack();
                
                static void nest();
                static void unnest();
                
                static bool is_nested_shallow();
                static bool is_nested_deep();
                static void destroy_chain();
            
            private:
                
                basura_t(void);
                basura_t(basura_t const&);
                basura_t(basura_t&&);
                basura_t& operator=(basura_t const&);
                basura_t& operator=(basura_t&&);
        };
        
    } /* namespace gil */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_GIL_GC_HPP_