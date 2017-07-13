
#include <libimread/ext/h5deets.hh>

namespace im {
    
    namespace detail {
        
        #pragma mark - Base HDF5 lifecycle manager (h5base) methods
        
        h5base::h5base(hid_t hid, releaser_f releaser)
            :m_hid(hid)
            ,m_releaser(releaser)
            {}
        
        h5base::~h5base() {
            /// call m_releaser on member HID:
            if (m_hid > 0) { m_releaser(m_hid); }
        }
        
        
        hid_t      h5base::get() const { return m_hid; }
        h5base::operator hid_t() const { return m_hid; }
        
        herr_t h5base::release() {
            if (m_hid < 0) { return -1; }
            herr_t out = m_releaser(m_hid);
            if (out > 0) { m_hid = -1; }
            return out;
        }
        
        bool h5base::operator==(h5base const& rhs) const {
            return m_hid ==  rhs.m_hid &&
             &m_releaser == &rhs.m_releaser;
            // return m_hid == rhs.m_hid;
        }
        
        bool h5base::operator!=(h5base const& rhs) const {
            return m_hid !=  rhs.m_hid ||
             &m_releaser != &rhs.m_releaser;
            // return m_hid != rhs.m_hid;
        }
        
        const h5base::releaser_f h5base::deref = [](hid_t hid) -> herr_t {
            return hid > 0 ? H5Idec_ref(hid) : -1;
        };
        
        /// NO-Op h5base releaser function:
        const h5base::releaser_f h5base::NOOp = [](hid_t hid) -> herr_t {
            return -1;
        };
        
        #pragma mark - HDF5 datatype wrapper (h5t_t) methods
        
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
        
        h5t_t& h5t_t::operator=(h5t_t const& other) {
            if (other != *this) {
                m_hid = H5Tcopy(other.m_hid);
                m_releaser = other.m_releaser;
            }
            return *this;
        }
        
        h5t_t& h5t_t::operator=(h5t_t&& other) noexcept {
            if (other != *this) {
                m_hid = std::move(other.m_hid);
                m_releaser = std::move(other.m_releaser);
                other.m_hid = -1;
            }
            return *this;
        }
        
        h5t_t::h5t_class_t h5t_t::cls() const {
            return H5Tget_class(m_hid);
        }
        
        h5t_t h5t_t::super() const {
            return h5t_t(H5Tget_super(m_hid));
        }
        
        #pragma mark - HDF5 attribute wrapper (h5a_t) methods
        
        h5a_t::h5a_t(hid_t parent_hid, std::size_t idx)
            :h5base(H5Aopen_idx(parent_hid, idx),
                    H5Aclose)
            ,m_memorytype(H5Aget_type(m_hid))
            ,m_parent_hid(parent_hid)
            ,m_dataspace_hid(H5Aget_space(m_hid))
            ,m_idx(idx)
            {
                /// increment refcount on the parent HID
                H5Iinc_ref(m_parent_hid);
                /// retrieve and store the attribute name
                char pbuffer[PATH_MAX] = { 0 };
                ssize_t length = H5Aget_name(parent_hid, sizeof(pbuffer),
                                                                pbuffer);
                if (length > 0) {
                    m_name = std::string(pbuffer, length);
                }
            }
        
        /// wrap existing attribute by name
        h5a_t::h5a_t(hid_t parent_hid, std::string const& name)
            :h5base(H5Aopen_name(parent_hid, name.c_str()),
                    H5Aclose)
            ,m_memorytype(H5Aget_type(m_hid))
            ,m_parent_hid(parent_hid)
            ,m_dataspace_hid(H5Aget_space(m_hid))
            ,m_name(name)
            {
                /// increment refcount on the parent HID
                H5Iinc_ref(m_parent_hid);
            }
        
        /// create entirely new attribute with name, space, and type
        h5a_t::h5a_t(hid_t parent_hid, std::string const& name,
                     hid_t dataspace_hid,
                     h5t_t datatype)
            :h5base(H5Acreate(parent_hid, name.c_str(),
                                          datatype.get(),
                                          dataspace_hid,
                                          H5P_DEFAULT, H5P_DEFAULT),
                    H5Aclose)
            ,m_memorytype(datatype)
            ,m_parent_hid(parent_hid)
            ,m_dataspace_hid(dataspace_hid)
            ,m_name(name)
            {
                /// increment refcount on the parent HID and dataspace
                H5Iinc_ref(m_parent_hid);
                H5Iinc_ref(m_dataspace_hid);
            }
        
        h5a_t::h5a_t(h5a_t&& other) noexcept
            :h5base(std::move(other.m_hid),
                    std::move(other.m_releaser))
            ,m_memorytype(std::move(other.m_memorytype))
            ,m_parent_hid(std::move(other.m_parent_hid))
            ,m_dataspace_hid(std::move(other.m_dataspace_hid))
            ,m_idx(other.m_idx)
            ,m_name(std::move(other.m_name))
            {
                /// increment refcount on the parent HID and dataspace
                H5Iinc_ref(m_parent_hid);
                H5Iinc_ref(m_dataspace_hid);
            }
        
        h5a_t::~h5a_t() {
            H5Idec_ref(m_parent_hid);
            H5Idec_ref(m_dataspace_hid);
        }
        
        herr_t h5a_t::read(void* buffer) const {
            return H5Aread(m_parent_hid,
                           m_memorytype.get(),
                           buffer);
        }
        
        herr_t h5a_t::read(void* buffer, h5t_t const& valuetype) const {
            return H5Aread(m_parent_hid,
                           valuetype.get(),
                           buffer);
        }
        
        herr_t h5a_t::write(const void* buffer) {
            return H5Awrite(m_parent_hid,
                            m_memorytype.get(),
                            buffer);
        }
        
        herr_t h5a_t::write(const void* buffer, h5t_t const& valuetype) {
            return H5Awrite(m_parent_hid,
                            valuetype.get(),
                            buffer);
        }
        
        h5t_t const& h5a_t::memorytype() const {
            return m_memorytype;
        }
        
        h5t_t const& h5a_t::memorytype(h5t_t const& n) {
            m_memorytype = n;
            return m_memorytype;
        }
        
        hid_t h5a_t::parent() const {
            return m_parent_hid;
        }
        
        hid_t h5a_t::dataspace() const {
            return m_dataspace_hid;
        }
        
        std::size_t h5a_t::idx() const {
            return m_idx;
        }
        
        std::string h5a_t::name() const {
            if (m_name == NULL_STR) {
                /// retrieve and store the attribute name
                char pbuffer[PATH_MAX] = { 0 };
                ssize_t length = H5Aget_name(m_parent_hid, sizeof(pbuffer),
                                                                  pbuffer);
                if (length > 0) {
                    m_name = std::string(pbuffer, length);
                }
            }
            return m_name;
        }
        
        // std::string h5a_t::name(std::string const& n) {
        //     m_name = n;
        //     return m_name;
        // }
        
    } /* namespace detail */
    
} /* namespace im */