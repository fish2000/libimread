
#include <libimread/ext/h5deets.hh>

namespace im {
    
    namespace detail {
        
        
        h5base::h5base(hid_t hid, releaser_f releaser)
            :m_hid(hid)
            ,m_releaser(releaser)
            {}
        
        h5base::~h5base() {
            /// call m_releaser on member HID:
            if (m_hid > 0) { m_releaser(m_hid); }
        }
        
        
        hid_t    h5base::get()   const { return m_hid; }
        h5base::operator hid_t() const { return m_hid; }
        
        herr_t h5base::release() {
            if (m_hid < 0) { return -1; }
            herr_t out = m_releaser(m_hid);
            if (out > 0) { m_hid = -1; }
            return out;
        }
        
        /// NO-Op h5base releaser function:
        const h5base::releaser_f h5base::NOOp = [](hid_t hid) -> herr_t { return -1; };
        
        h5t_t::h5t_t(hid_t hid)
            :h5base(H5Tcopy(hid),
                    H5Tclose)
            {}
        
        h5t_t::h5t_t(h5t_t::h5t_class_t cls, std::size_t size)
            :h5base(H5Tcreate(cls, size),
                    H5Tclose)
            {}
        
        h5t_t::h5t_t(h5t_t const& other)
            :h5base(H5Tcopy(other.m_hid),
                            other.m_releaser)
            {}
        
        h5t_t::h5t_t(h5t_t&& other) noexcept
            :h5base(std::move(other.m_hid),
                    std::move(other.m_releaser))
            { other.m_hid = -1; }
        
        h5t_t::h5t_class_t h5t_t::cls() const {
            return H5Tget_class(m_hid);
        }
        
        h5t_t h5t_t::super() const {
            return h5t_t(H5Tget_super(m_hid));
        }
        
        
        
    } /* namespace detail */
    
} /* namespace im */