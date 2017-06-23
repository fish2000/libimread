/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_DIRECTORY_H_
#define LIBIMREAD_EXT_FILESYSTEM_DIRECTORY_H_

#include <string>
#include <mutex>
#include <stack>
#include <unistd.h>

namespace filesystem {
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        using pathstack_t = std::stack<path>;
    }
    
    struct switchdir final {
        
        /// Change working directory temporarily with RAII while
        /// holding a process-unique lock during the switchover.
        ///
        ///     // assume we ran as "cd /usr/local/bin && ./yodogg"
        ///     using filesystem::path;
        ///     path new_directory = path("/usr/local/var/myshit");
        ///     // most list() calls yield pathvec_t, a std::vector<filesystem::path>
        ///     filesystem::detail::pathvec_t stuff;
        ///     {
        ///         filesystem::switchdir switch(new_directory); // threads block here serially
        ///         assert(path::cwd() == "/usr/local/var/myshit");
        ///         assert(switchdir.from() == "/usr/local/bin");
        ///         stuff = path::cwd().list();
        ///     }
        ///     // scope exit destroys the `switch` instance --
        ///     // restoring the process-wide previous working directory
        ///     // and unlocking the global `filesystem::switchdir` mutex
        ///     assert(path::cwd() == "/usr/local/bin");
        ///     std::cout << "We found " << stuff.size()
        ///               << " items inside  " new_directory << "..."
        ///               << std::endl;
        ///     std::cout << "We're currently working out of " << path::cwd() << ""
        ///               << std::endl;
        ///
        /// This avoids a slew of the race conditions you are risking
        /// whenever you start shooting off naked ::chdir() calls --
        /// including wierd results from APIs like the POSIX filesystem calls
        /// (e.g. ::glob() and ::readdir(), both of which path.cpp leverages).
        /// Those older C-string-based interfaces are generous with
        /// semantic vagaries, and can behave in ways that make irritating
        /// use of, or assumptions about, the process' current working directory.
        /// ... and so yeah: "Block before chdir()" is the new "Use a condom".
        
        explicit switchdir(path const& nd)
            :olddir(path::cwd().str())
            ,newdir(nd.make_absolute().str())
            {
                mute.lock();
                ::chdir(newdir.c_str());
            }
        
        path from() const {
            return path(olddir);
        }
        
        ~switchdir() {
            ::chdir(olddir.c_str());
            mute.unlock();
        }
        
        private:
            switchdir(void);
            switchdir(switchdir const&);
            switchdir(switchdir&&);
            switchdir& operator=(switchdir const&);
            switchdir& operator=(switchdir&&);
            
        private:
            static std::mutex mute;         /// declaration not definition
            mutable std::string olddir;
            mutable std::string newdir;
    };
    
    struct workingdir final {
        
        /// Change working directory multiple times with a RAII idiom,
        /// using a process-unique recursive lock to maintain a stack of
        /// previously occupied directories.
        /// rewinding automatically to the previous originating directory
        /// on scope exit.
        
        explicit workingdir(path&& nd)
            { push(std::forward<path>(nd)); }
        
        path from() const {
            return path(top());
        }
        
        ~workingdir() {
            pop();
        }
        
        static void push(path&& nd) {
            if (nd == dstack.top()) { return; }
            mute.lock();
            dstack.push(path::cwd());
            ::chdir(nd.c_str());
        }
        
        static void pop() {
            if (dstack.empty()) { return; }
            mute.unlock();
            ::chdir(dstack.top().c_str());
            dstack.pop();
        }
        
        static path const& top() {
            return dstack.empty() ? empty : dstack.top();
        }
        
        private:
            workingdir(void);
            workingdir(workingdir const&);
            workingdir(workingdir&&);
            workingdir& operator=(workingdir const&);
            workingdir& operator=(workingdir&&);
            
        private:
            static std::recursive_mutex mute;   /// declaration not definition
            static detail::pathstack_t dstack;  /// declaration not definition
            static const path empty;            /// declaration not definition
    };
    
}

#endif /// LIBIMREAD_EXT_FILESYSTEM_DIRECTORY_H_