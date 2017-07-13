
#include <libimread/ext/h5deets.hh>
// #include <libimread/errors.hh>

namespace im {
    
    namespace detail {
        
        #pragma mark - Base HDF5 lifecycle manager (h5base) methods
        
        h5base::h5base(hid_t hid, releaser_f releaser)
            :m_hid(hid)
            ,m_releaser(releaser)
            {}
        
        h5base::~h5base() {
            /// call m_releaser on member HID:
            // WTF("[h5base] About to call releaser function on primary HID ", m_hid,
            //     "[h5base] Primary HID has refcount: ", H5Iget_ref(m_hid));
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
        }
        
        bool h5base::operator!=(h5base const& rhs) const {
            return m_hid !=  rhs.m_hid ||
             &m_releaser != &rhs.m_releaser;
        }
        
        int h5base::incref() {
            return m_hid > 0 ? H5Iinc_ref(m_hid) : -1;
        }
        
        int h5base::decref() {
            return m_hid > 0 ? H5Idec_ref(m_hid) : -1;
        }
        
        /// generic reference-count-based releaser function:
        const h5base::releaser_f h5base::unref = [](hid_t hid) -> herr_t {
            return hid > 0 ? H5Idec_ref(hid) : -1;
        };
        
        /// NO-Op releaser function:
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
        
        #pragma mark - HDF5 attribute-dataspace wrapper (attspace_t) methods
        
        attspace_t::attspace_t(void)
            :h5base(-1, H5Sclose)
            {}
        
        attspace_t::attspace_t(hid_t hid)
            :h5base(hid, H5Sclose)
            {}
        
        attspace_t attspace_t::scalar() {
            return attspace_t(H5Screate(H5S_SCALAR));
        }
        
        attspace_t attspace_t::simple() {
            return attspace_t(H5Screate(H5S_SIMPLE));
        }
        
        attspace_t::attspace_t(attspace_t const& other)
            :h5base(other.m_hid,
                    other.m_releaser)
            {
                incref();
            }
        
        attspace_t::attspace_t(attspace_t&& other) noexcept
            :h5base(std::move(other.m_hid),
                    std::move(other.m_releaser))
            {
                incref();
            }
        
        attspace_t& attspace_t::operator=(attspace_t const& other) {
            if (other != *this) {
                m_hid = other.m_hid;
                m_releaser = other.m_releaser;
                incref();
            }
            return *this;
        }
        
        attspace_t& attspace_t::operator=(attspace_t&& other) noexcept {
            if (other != *this) {
                m_hid = std::move(other.m_hid);
                m_releaser = std::move(other.m_releaser);
                incref();
            }
            return *this;
        }
        
        #pragma mark - HDF5 attribute wrapper (h5a_t) methods
        
        h5a_t::h5a_t(hid_t parent_hid, std::size_t idx)
            :h5base(H5Aopen_idx(parent_hid, idx),
                    H5Aclose)
            ,m_parent_hid(parent_hid)
            ,m_memorytype(H5Aget_type(m_hid))
            ,m_dataspace(H5Aget_space(m_hid))
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
            ,m_parent_hid(parent_hid)
            ,m_memorytype(H5Aget_type(m_hid))
            ,m_dataspace(H5Aget_space(m_hid))
            ,m_name(name)
            {
                /// increment refcount on the parent HID
                H5Iinc_ref(m_parent_hid);
            }
        
        /// create entirely new attribute with name, space, and type
        h5a_t::h5a_t(hid_t parent_hid, std::string const& name,
                     hid_t dataspace_hid,
                     h5t_t datatype)
            :h5base(H5Acreate2(parent_hid, name.c_str(),
                                           datatype.get(),
                                           dataspace_hid,
                                           H5P_DEFAULT, H5P_DEFAULT),
                    H5Aclose)
            ,m_parent_hid(parent_hid)
            ,m_memorytype(datatype)
            ,m_dataspace(dataspace_hid)
            ,m_name(name)
            {
                /// increment refcount on the parent HID and dataspace
                H5Iinc_ref(m_parent_hid);
                m_dataspace.incref();
            }
        
        h5a_t::h5a_t(h5a_t&& other) noexcept
            :h5base(std::move(other.m_hid),
                    std::move(other.m_releaser))
            ,m_parent_hid(std::move(other.m_parent_hid))
            ,m_memorytype(std::move(other.m_memorytype))
            ,m_dataspace(std::move(other.m_dataspace))
            ,m_idx(other.m_idx)
            ,m_name(std::move(other.m_name))
            {
                /// increment refcount on the parent HID
                H5Iinc_ref(m_parent_hid);
            }
        
        h5a_t::~h5a_t() {
            H5Idec_ref(m_parent_hid);
        }
        
        herr_t h5a_t::read(void* buffer) const {
            return H5Aread(m_hid,
                           m_memorytype.get(),
                           buffer);
        }
        
        herr_t h5a_t::read(void* buffer, h5t_t const& valuetype) const {
            return H5Aread(m_hid,
                           valuetype.get(),
                           buffer);
        }
        
        herr_t h5a_t::write(const void* buffer) {
            return H5Awrite(m_hid,
                            m_memorytype.get(),
                            buffer);
        }
        
        herr_t h5a_t::write(const void* buffer, h5t_t const& valuetype) {
            return H5Awrite(m_hid,
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
        
        attspace_t h5a_t::dataspace() const {
            attspace_t out(m_dataspace);
            return out;
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
        
    } /* namespace detail */
    
} /* namespace im */