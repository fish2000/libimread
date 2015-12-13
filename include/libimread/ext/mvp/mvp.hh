#include <cstdlib>
#include <cstring>
#include <cmath>

#include <array>
#include <deque>
#include <tuple>
#include <random>
#include <vector>
#include <string>
#include <memory>
#include <locale>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

namespace detail {

    static const float kComparatorBadArguments = -1.0f;
    static const float kComparatorUnequalDataVectorLengths = -2.0f;
    
    template <typename PathType,
              typename = std::enable_if_t<
                         std::is_floating_point<PathType>::value>> inline
    bool isnan(PathType x) {
        PathType var = x;
        return var != var;
    }
    
    template <typename PathType,
              typename = std::enable_if_t<
                         std::is_integral<PathType>::value>,
              typename = void> inline
    bool isnan(PathType x) {
        return std::isnan(x);
    }
    
} /* namespace detail */


template <typename PathType = float,
          std::size_t PathLength = 5>
class Base {
    public:
        static constexpr std::size_t PL = PathLength;
        
        using path_t = PathType;
        using pilist_t = std::initializer_list<path_t>;
        using pathlist_t = std::array<path_t, PL>;
        using pathvec_t = std::vector<path_t>;
        
        virtual ~Base() {
            // std::cout << "\t*** " << "Base::~Base()" << std::endl;
        }
};

/// Curiouser And Curiouser!
template <typename DataPoint>
class DataPointBase : Base<> {
    
    public:
        using Base<>::PL;
        using Base<>::path_t;
        using Base<>::pilist_t;
        using Base<>::pathlist_t;
        using Base<>::pathvec_t;
        using pointer_t = std::add_pointer_t<DataPoint>;
        using pointvec_t = std::vector<pointer_t>;
        using comparator_t = std::function<path_t(pointer_t, pointer_t)>;
    
    std::string name;
    pathlist_t paths;
    comparator_t comparator;
    
    public:
        
        DataPointBase()
            :name("")
            ,paths{{ 0.0f }}
            ,comparator(DataPoint::default_comparator)
            {}
        
        DataPointBase(const DataPointBase& other)
            :name(other.name)
            ,paths(other.paths)
            ,comparator(other.comparator)
            {}
        
        DataPointBase(DataPointBase&& other)
            :name(std::move(other.name))
            ,paths(std::move(other.paths))
            ,comparator(std::move(other.comparator))
            {}
        
        void set_paths(pilist_t pilist) {
            std::size_t idx = 0;
            for (auto it = pilist.begin();
                 it != pilist.end() && idx < PL;
                 ++it) { paths[idx] = *it;
                         ++idx; }
        }
        void set_paths(const pathlist_t& pathlist) {
            paths = pathlist;
        }
        
        inline path_t compare(pointer_t other) {
            std::cout << "\t*** " << "DataPointBase::compare()" << std::endl;
            return comparator(static_cast<pointer_t>(this), other);
        }
        
        bool distance_range(pointvec_t& points, std::size_t lvl = 0) {
            std::cout << "\t*** " << "DataPointBase::distance_range()" << std::endl;
            if (points.empty()) { return false; }
            std::size_t i, im = points.size();
            path_t d;
            for (i = 0; i < im; i++) {
                d = comparator(static_cast<pointer_t>(this),
                               points[i]);
                if (detail::isnan(d) || d < 0.0f) { return false; }
                if (lvl < PL) {
                    points[i]->paths[lvl] = d;
                }
            }
            return true;
        }
        
        bool splits(const pointvec_t& points, pathvec_t& M,
                                              std::size_t offset = 0) {
            std::cout << "\t*** " << "DataPointBase::splits()" << std::endl;
            if (points.empty()) { return false; }
            
            pathvec_t distances(points.size(), 0.0f);
            std::transform(points.begin(), points.end(),
                           distances.begin(),
                           [&](pointer_t p) {
                return comparator(static_cast<pointer_t>(this), p);
            });
            
            path_t tmp;
            std::size_t i, j, min_pos, idx,
                        im = points.size(),
                        Mm = M.size();
            
            for (i = 0; i < im-1; i++) {
                min_pos = i;
                for (j = i+1; j < im; j++) {
                    if (distances[j] < distances[min_pos]) {
                        min_pos = j;
                    }
                }
                if (min_pos != i) {
                    tmp = distances[min_pos];
                    distances[min_pos] = distances[i];
                    distances[i] = tmp;
                }
            }
            
            for (i = 0; i < Mm; i++) {
                idx = (i+1)*im/(Mm+1);
                if (idx <= 0) { idx = 0; }
                if (idx >= im) { idx = im-1; }
                M[i] = distances[idx];
            }
            
            return true;
        }
};

template <typename DataType,
          typename PathType = float,
          std::size_t DataLength = 1, /// dummy param
          std::size_t PathLength = 5>
class Datum : public DataPointBase<Datum<DataType, PathType,
                                         DataLength, PathLength>> {
    
    struct ComparisonOperator { 
        PathType operator()(Datum* d1, Datum* d2) {
            std::cout << "\t--- " << "Datum::ComparisonOperator::operator()" << std::endl;
            if (d1->datum == 0 || d2->datum == 0) {
                return detail::kComparatorBadArguments;
            }
            return (PathType)std::abs((int)d1->datum - (int)d2->datum);
        }
    };
    
    public:
        using DPBase = DataPointBase<Datum<DataType, PathType,
                                           DataLength, PathLength>>;
        using default_comparator_t = ComparisonOperator;
        using data_t = DataType;
        
        static constexpr std::size_t DL = 1; /// hardcoded
        static constexpr default_comparator_t default_comparator{};
        data_t datum;
        
        Datum() : DPBase()
            ,datum(0)
            {}
        
        explicit Datum(data_t d) : DPBase()
            ,datum(d)
            {}
        
        Datum(const Datum& other) : DPBase(other)
            ,datum(other.datum)
            {}
        
        Datum(Datum&& other) : DPBase(other)
            ,datum(std::move(other.datum))
            {}
        
        void assign(data_t d) {
            datum = d;
        }
        void assign(data_t* dptr, std::size_t idx = 0) {
            datum = dptr[idx];
        }
        
        inline DataType* data(std::size_t idx = 0) {
            return &datum;
        }
        
        template <typename CastType> inline
        CastType* data_as(std::size_t idx = 0) {
            return static_cast<CastType*>(&datum);
        }
        
        inline bool operator==(const Datum& rhs) { return datum == rhs.datum; }
        inline bool operator!=(const Datum& rhs) { return datum != rhs.datum; }
        inline bool operator<(const Datum& rhs)  { return datum < rhs.datum; }
        inline bool operator>(const Datum& rhs)  { return datum > rhs.datum; }
        inline bool operator<=(const Datum& rhs) { return datum <= rhs.datum; }
        inline bool operator>=(const Datum& rhs) { return datum >= rhs.datum; }
        
};

template <typename DataType,
          typename PathType = float,
          std::size_t DataLength = 10, /// this is arbitrary
          std::size_t PathLength = 5>
class Vector : public DataPointBase<Vector<DataType, PathType,
                                           DataLength, PathLength>> {
        
    struct ComparisonOperator {
        PathType operator()(Vector* d1, Vector* d2) {
            // std::cout << "\t--- " << "Vector::ComparisonOperator::operator()" << std::endl;
            if (!d1 || !d2) {
                // std::cout << "\t--- " << "Vector::ComparisonOperator::operator() [NULL POINTER ARGS]" << std::endl;
                return detail::kComparatorBadArguments;
            }
            if (d1->datavec.empty() || d2->datavec.empty()) {
                std::cout << "\t--- " << "Vector::ComparisonOperator::operator() [EMPTY DATA VECTORS]" << std::endl;
                return detail::kComparatorBadArguments;
            }
            if (d1->datavec.size() != d2->datavec.size()) {
                std::cout << "\t--- " << "Vector::ComparisonOperator::operator() [UNEQUAL DATA VECTORS]" << std::endl;
                return detail::kComparatorUnequalDataVectorLengths;
            }
            std::size_t idx, sum = 0, max = d1->datavec.size();
            for (idx = 0; idx < max; ++idx) {
                sum += std::abs((int)d1->datavec[idx] - (int)d2->datavec[idx]);
            }
            // std::cout << "\t--- " << "Vector::ComparisonOperator::operator() [RETURNING SUCCESSFULLY]" << std::endl;
            return (PathType)sum/(PathType)max;
        }
    };
    
    public:
        using DPBase = DataPointBase<Vector<DataType, PathType,
                                            DataLength, PathLength>>;
        using default_comparator_t = ComparisonOperator;
        using data_t = DataType;
        using ilist_t = std::initializer_list<data_t>;
        using datavec_t = std::vector<data_t>;
        
        static constexpr std::size_t DL = DataLength;
        static constexpr std::size_t DS = sizeof(DataType);
        static constexpr default_comparator_t default_comparator{};
        datavec_t datavec;
        
        Vector() : DPBase()
            ,datavec(DL, 0)
            {}
        
        explicit Vector(ilist_t ilist) : DPBase()
            ,datavec(ilist)
            {}
        
        Vector(const Vector& other) : DPBase(other)
            ,datavec(other.datavec)
            {}
        
        Vector(Vector&& other) : DPBase(other)
            ,datavec(std::move(other.datavec))
            {}
        
        void assign(ilist_t ilist) {
            datavec = datavec_t(ilist);
        }
        void assign(data_t* dptr, std::size_t idx = DL) {
            datavec.reserve(idx);
            std::memmove((void*)datavec.data(),
                         (const void*)dptr, DS*idx);
        }
        
        inline DataType* data(std::size_t idx = 0) {
            return &datavec[idx];
        }
        
        template <typename CastType> inline
        CastType* data_as(std::size_t idx = 0) {
            return static_cast<CastType*>(data(idx));
        }
        
        template <typename BinaryPredicate> inline
        bool binary_op(const Vector& rhs,
                       BinaryPredicate predicate = BinaryPredicate()) {
            if (rhs.datavec.empty()) { return datavec.empty(); }
            if (datavec.empty()) { return false; }
            std::cout << "\t*** " << "Vector::binary_op()" << std::endl;
            std::cout << "\t*** " << "Vector::binary_op() [datavec.size() = " << datavec.size() << "]" << std::endl;
            std::cout << "\t*** " << "Vector::binary_op() [rhs.datavec.size() = " << rhs.datavec.size() << "]" << std::endl;
            if (datavec.size() != rhs.datavec.size()) { return false; }
            std::cout << "\t*** " << "Vector::binary_op() [RETURNING WITH ITERATOR]" << std::endl;
            return std::equal(std::begin(datavec),      std::end(datavec),
                              std::begin(rhs.datavec),  std::end(rhs.datavec),
                              predicate);
        }
        
        inline bool operator==(const Vector& rhs) { return binary_op<std::equal_to<data_t>>(rhs); }
        inline bool operator!=(const Vector& rhs) { return binary_op<std::not_equal_to<data_t>>(rhs); }
        inline bool operator<(const Vector& rhs)  { return binary_op<std::less<data_t>>(rhs); }
        inline bool operator>(const Vector& rhs)  { return binary_op<std::greater<data_t>>(rhs); }
        inline bool operator<=(const Vector& rhs) { return binary_op<std::less_equal<data_t>>(rhs); }
        inline bool operator>=(const Vector& rhs) { return binary_op<std::greater_equal<data_t>>(rhs); }
};

template <typename DataType,
          template <typename, typename, std::size_t, std::size_t>
          class DataPointTemplate  = Datum,
          std::size_t DataLength   = 1,
          std::size_t BranchFactor = 2,
          std::size_t PathLength   = 5,
          std::size_t LeafCap      = 25,
          typename PathType        = float,
          typename DataPoint       = DataPointTemplate<DataType, PathType,
                                                       DataLength, PathLength>,
          std::size_t FanOut       = BranchFactor*BranchFactor,
          std::size_t LengthM1     = BranchFactor-1,
          std::size_t LeafCapHigh  = LeafCap+2>
class Tree {
    
    public:
        static constexpr std::size_t BF = BranchFactor;
        static constexpr std::size_t PL = PathLength;
        static constexpr std::size_t DL = DataLength;
        static constexpr std::size_t LCp = LeafCap;
        static constexpr std::size_t FnO = FanOut;
        static constexpr std::size_t LM1 = LengthM1;
        static constexpr std::size_t LM2 = LengthM1 - 1;
        static constexpr std::size_t LCH = LeafCapHigh;
        
        using data_t = DataType;
        using path_t = PathType;
        using datapoint_t = DataPoint;
        using datapointer_t = std::add_pointer_t<datapoint_t>;
        using pointvec_t = std::vector<datapointer_t>;
        using pathvec_t = std::vector<path_t>;
        using Comparator = typename datapoint_t::comparator_t;
        using comparator_t = typename datapoint_t::comparator_t;
        
        enum NodeType : std::size_t {
            ABSTRACT = 0,
            INTERNAL = 1,
            LEAF };
        
        struct InternalTag {};
        struct LeafTag {};
    
    private:
        struct NodeBase {
            NodeType nodetype;
            datapointer_t sv1;
            datapointer_t sv2;
            NodeBase(NodeType ntype = NodeType::ABSTRACT)
                :nodetype(ntype)
                ,sv1(nullptr), sv2(nullptr)
                {
                    std::cout << "\t*** " << "\t*** " << "NodeBase::NodeBase()" << std::endl;
                }
            inline NodeType type() const { return nodetype; }
            inline bool isInternal() const { return nodetype == NodeType::INTERNAL; }
            inline bool isLeaf() const { return nodetype == NodeType::LEAF; }
            void destroy() { if (sv1 != nullptr) delete sv1;
                             if (sv2 != nullptr) delete sv2; }
            virtual void clear(int lvl = 0) { destroy(); }
            virtual ~NodeBase() {}
        };
        struct InternalNode : virtual public NodeBase {
            pathvec_t M1, M2;
            std::array<std::add_pointer_t<NodeBase>, FanOut> children;
            InternalNode() : NodeBase(NodeType::INTERNAL)
                ,M1(LengthM1), M2(BranchFactor)
                ,children{}
                {
                    std::cout << "\t*** " << "\t*** " << "InternalNode::InternalNode()" << std::endl;
                }
            virtual void clear(int lvl = 0) {
                for (auto* node : children) {
                    node->clear(lvl+1);
                }
                NodeBase::destroy();
            }
        };
        struct LeafNode : virtual public NodeBase {
            pathvec_t d1, d2;
            pointvec_t points;
            LeafNode() : NodeBase(NodeType::LEAF)
                ,d1(LeafCap), d2(LeafCap)
                ,points(LeafCap)
                {
                    std::cout << "\t*** " << "\t*** " << "LeafNode::LeafNode()" << std::endl;
                }
        };
        struct Node : public InternalNode, public LeafNode {
            explicit Node(InternalTag x)
                :InternalNode()
                {
                    std::cout << "\t*** " << "\t*** " << "Node::Node(InternalTag)" << std::endl;
                }
            explicit Node(LeafTag x)
                :LeafNode()
                {
                    std::cout << "\t*** " << "\t*** " << "Node::Node(LeafTag)" << std::endl;
                }
        };
    
    protected:
        using node_t = Node;
        using nodepointer_t = std::add_pointer_t<node_t>;
        using idx_t = std::tuple<std::size_t, std::size_t>;
        using vantagepoints_t = std::tuple<datapointer_t, datapointer_t>;
        using histogram_t = std::array<pointvec_t, BF>;
        int descriptor;
        int knearest;
        nodepointer_t top;
        comparator_t comparator;
        
        histogram_t sort_points(datapoint_t* vantagepoint,
                                const pointvec_t& points,
                                const vantagepoints_t& boundaries,
                                const pathvec_t& pivots) {
            std::cout << "\t*** " << "Tree<...>::sort_points()" << std::endl;
            
            histogram_t bins{};
            path_t d;
            std::size_t i, k,
                        im = points.size();
            
            std::cout << "\t*** " << "Tree<...>::sort_points() [ENTERING LOOP]" << std::endl;
            for (i = 0; i < im; i++) {
                /// WHY THE FUCK DOES THIS SEGFAULT
                /// ... I mean OK I confess it is totally Rube Goldberg
                /// but COME ON dogg
                // if (*points[i] == *std::get<0>(boundaries)) { continue; }
                // if (*points[i] == *std::get<1>(boundaries)) { continue; }
                
                d = comparator(vantagepoint, points[i]);
                for (k = 0; k < LM1; k++) {
                    if (d <= pivots[k]) {
                        bins[k].push_back(points[i]);
                        break;
                    }
                }
                if (d > pivots[LM2]) {
                    bins[LM1].push_back(points[i]);
                }
            }
            
            return bins;
        }
        
        idx_t vantage_indexes(const pointvec_t& points) {
            std::cout << "\t*** " << "Tree<...>::vantage_indexes()" << std::endl;
            std::size_t sv1pos = (points.size() >= 1) ? 0 : -1;
            std::size_t sv2pos = -1;
            PathType maxDist = 0.0f, d;
            std::size_t i, j,
                        im = points.size(),
                        jm = points.size();
            for (i = 0; i < im; i++) {
                for (j = i+1; j < jm; j++) {
                    d = comparator(points[i], points[j]);
                    // if (d < 0) {
                    //     std::cout << "\t*** " << "Tree<...>::vantage_indexes() [COMPARATOR ERROR]" << std::endl;
                    // }
                    if (d > maxDist) {
                        maxDist = d;
                        sv1pos = i;
                        sv2pos = j;
                    }
                }
            }
            return std::make_tuple(sv1pos, sv2pos);
        }
        
        vantagepoints_t vantage_points(const pointvec_t& points) {
            std::cout << "\t*** " << "Tree<...>::vantage_points()" << std::endl;
            std::size_t sv1pos, sv2pos;
            std::tie(sv1pos, sv2pos) = vantage_indexes(points);
            return std::make_tuple(points[sv1pos], points[sv2pos]);
        }
        
        nodepointer_t addNode(nodepointer_t node,
                              pointvec_t& points,
                              std::size_t lvl = 0) {
            std::cout << "\t*** " << "Tree<...>::addNode()" << std::endl;
            if (points.empty()) {
                return node;
            }
            
            nodepointer_t newNode = node;
            std::size_t i, count = 0,
                          pcount = points.size(),
                          ncount = 0,
                             idx = 0,
                               j = 0;
            
            if (newNode == NULL || newNode == nullptr) {
                /// create new node
                if (pcount <= LCH) {
                    /// create leaf node
                    newNode = new Node(LeafTag{});
                    std::tie(newNode->sv1, newNode->sv2) = vantage_points(points);
                    newNode->sv1->distance_range(points, lvl);
                    newNode->sv2->distance_range(points, lvl+1);
                    
                    for (i = 0; i < pcount; i++) {
                        if (points[i] && newNode->sv1 && newNode->sv2) {
                            if (*points[i] == *(newNode->sv1)) { continue; }
                            if (*points[i] == *(newNode->sv2)) { continue; }
                        }
                        newNode->d1[count] = newNode->sv1->compare(points[i]);
                        newNode->d2[count] = newNode->sv2->compare(points[i]);
                        newNode->points.push_back(points[i]);
                        count++;
                    }
                } else {
                    /// create internal node
                    newNode = new Node(InternalTag{});
                    std::tie(newNode->sv1, newNode->sv2) = vantage_points(points);
                    newNode->sv1->distance_range(points, lvl);
                    newNode->sv1->splits(points, newNode->M1);
                    histogram_t bins = sort_points(newNode->sv1, points,
                                                   std::make_tuple(newNode->sv1,
                                                                   newNode->sv2),
                                                   newNode->M1);
                    /// loop [0..BF) with bins
                    for (i = 0; i < BF; i++) {
                        newNode->sv2->distance_range(bins[i], lvl+1);
                        newNode->sv2->splits(bins[i], newNode->M2);
                    }
                }
            } else {
                /// node already exists
                if (newNode->isLeaf()) {
                    /// node is a leaf node
                    if (newNode->points.size() + pcount <= LCp) {
                        /// add points into leaf ("plenty of room")
                        newNode->sv1->distance_range(points, lvl);
                        std::size_t pos = 0;
                        if (newNode->sv2 == nullptr) {
                            newNode->sv2 = points[0];
                            pos = 1;
                        }
                        newNode->sv2->distance_range(points, lvl);
                        count = newNode->points.size();
                        for (; pos < pcount; pos++) {
                            newNode->d1[count] = comparator(points[pos],
                                                            newNode->sv1);
                            newNode->d2[count] = comparator(points[pos],
                                                            newNode->sv2);
                            newNode->points[count++] = points[pos];
                        }
                        newNode->points.reserve(count);
                    } else {
                        /// not enough room in current leaf --
                        /// create new node
                        ncount = newNode->points.size() + pcount;
                        if (newNode->sv1) { ncount++; }
                        if (newNode->sv2) { ncount++; }
                        
                        pointvec_t temporary(ncount);
                        i = idx = 0;
                        if (newNode->sv1) { temporary.push_back(newNode->sv1); }
                        if (newNode->sv2) { temporary.push_back(newNode->sv2); }
                        /// NOTE TO SELF: redo these two forloopies
                        /// ...WITH ITERATORS!
                        for (; i < newNode->points.size(); i++) {
                            temporary.push_back(newNode->points[i]);
                        }
                        for (i = 0; i < pcount; i++) {
                            temporary.push_back(points[i]);
                        }
                        
                        // nodepointer_t oldNode = newNode;
                        /// Need to DESTROY newNode here
                        newNode = addNode(nullptr, temporary, lvl);
                        
                        /// temporary contains no new allocations --
                        /// it can safely destruct on scope exit:
                    }
                } else {
                    /// node is an internal node --
                    /// must recurse on subnodes
                    newNode->sv1->distance_range(points, lvl);
                    histogram_t bins = sort_points(newNode->sv1, points,
                                                   std::make_tuple(nullptr, nullptr),
                                                   newNode->M1);
                    /// loop [0..BF) with bins
                    for (i = 0; i < BF; i++) {
                        newNode->sv2->distance_range(bins[i], lvl+1);
                        
                    }
                }
            }
            
            /// all is well - reset the tree's root node
            return newNode;
        }
        
    
    public:
        
        Tree()
            :descriptor(0)
            ,knearest(0)
            ,top(nullptr)
            ,comparator(DataPoint::default_comparator)
            {}
        
        void clear() {}
        virtual ~Tree() {
            if (top) { delete top; }
        }
        
        void add(pointvec_t& points) {
            std::cout << "\t*** " << "Tree<...>::add()" << std::endl;
            top = addNode(top, points);
        }
        
        // TODOODOO:
        // add()
        // retrieve()
        
};

using BasicTree = Tree<uint8_t>;
using VectorTree = Tree<uint8_t, Vector, 10>;
using VectorPoint = typename VectorTree::datapoint_t;
using PointVector = typename VectorTree::pointvec_t;

template <typename DataType>
using Randomizer = std::independent_bits_engine<
                   std::default_random_engine,
                   sizeof(DataType)*8, DataType>;

using wc_t = std::ctype<wchar_t>;

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vector) {
    if (vector.empty()) { return out; }
    auto const& facet = std::use_facet<wc_t>(std::locale());
    out << "<0x" << std::hex;
    std::transform(vector.begin(), vector.end(),
                std::ostream_iterator<T>(out, ""),
                [&facet](T const& t) {
        return facet.narrow(t, t/4);
    });
    out << std::dec << "\b\b>";
    return out;
}

/// Yo, so you are 'not allowed' to declare `main()` as static...
/// but FYI, the following exemplary extemporaneousness wherein
/// you extern-"C" that sucker inside an anonymous namespace
/// evidently serves to accomplish the same basic idea --
/// which, when this happens in a header file, is some handy shit

namespace {
    extern "C" {
        int main(void) {
            
            #define A_BUNCH 30
            
            std::locale::global(std::locale("en_US"));
            std::cout.imbue(std::locale());
            std::cerr.imbue(std::locale());
            
            std::cerr << "TCZ-MVPTREE 0.1.0 (DoggNodeType 0.6.3, A_BUNCH = "
                      << A_BUNCH << ") :: Starting Up"
                      << std::endl;
            
            /// make up a bunch of VectorPoints with random data
            using data_t = typename VectorPoint::data_t;
            using randomizer_t = Randomizer<data_t>;
            PointVector pv(A_BUNCH);
            randomizer_t randomizer;
            
            std::cout << "\t" << "Heap-allocating "
                              << A_BUNCH
                              << " random VectorPoint* instances"
                      << std::endl;
            
            for (int idx = 0; idx < A_BUNCH; idx++) {
                VectorPoint* p = new VectorPoint();
                std::generate(std::begin(p->datavec),
                              std::end(p->datavec),
                              std::ref(randomizer));
                std::cout << "\t*** " << "Allocated point #" << idx
                                      << " of " << A_BUNCH
                                      << " with data: " << p->datavec
                                      << ""
                          << std::endl;
                pv.push_back(p);
            }
            
            std::cout << "\t" << "Stack-allocating VectorTree"
                      << std::endl;
            
            BasicTree btree;
            VectorTree vtree;
            
            std::cout << "\t" << "Passing heap VectorPoint container to VectorTree stack instance"
                      << std::endl;
            
            vtree.add(pv);
            
            std::cout << "\t" << "Deleting VectorPoint* instances from heap"
                      << std::endl;
            
            /// delete random VectorPoints
            std::for_each(pv.begin(), pv.end(),
                      [=](VectorPoint* p) { delete p; });
            
            std::cerr << "TCZ-MVPTREE 0.1.0 (DoggNodeType 0.6.3) :: Return To Zero"
                      << std::endl;
            
            return 0;
            
        } /* main() */
    } /* extern "C" */
} /* namespace (anon.) */
