#include <cstdlib>
#include <cmath>

#include <array>
#include <deque>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

static const float kComparatorBadArguments = -1.0f;
static const float kComparatorUnequalDataVectorLengths = -2.0f;

namespace detail {
    
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
        
        virtual ~Base() {}
};

/// Curiouser And Curiouser!
template <typename DataPoint>
class DataPointBase : Base<> {
    
    // static constexpr std::size_t DS = sizeof(typename DataPoint::data_t);
    public:
        using Base<>::PL;
        using Base<>::path_t;
        using Base<>::pilist_t;
        using Base<>::pathlist_t;
        using Base<>::pathvec_t;
        using pointer_t = std::add_pointer_t<DataPoint>;
        using pointvec_t = std::vector<pointer_t>;
        using comparator_t = std::function<path_t(const DataPoint&,
                                                  const DataPoint&)>;
    
    std::string name;
    pathlist_t paths;
    comparator_t comparator;
    
    public:
        
        DataPointBase()
            :name("")
            ,paths{ 0.0f }
            ,comparator(DataPoint::comparator_t())
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
            int idx = 0;
            for (auto it = pilist.begin();
                 it != pilist.end() && idx < PL;
                 ++it) { paths[idx] = *it;
                         ++idx; }
        }
        void set_paths(const pathlist_t& pathlist) {
            paths = pathlist;
        }
        
        inline path_t compare(const DataPoint& other) {
            return comparator(*this, other);
        }
        
        bool distance_range(pointvec_t& points, int lvl = 0) {
            if (points.empty()) { return false; }
            std::size_t i, im = points.size();
            path_t d;
            for (i = 0; i < im; i++) {
                d = comparator(*this, points[i]);
                if (detail::isnan(d) || d < 0.0f) { return false; }
                if (lvl < PL) {
                    points[i]->paths[lvl] = d;
                }
            }
            return true;
        }
        
        bool splits(const pointvec_t& points, pathvec_t& M) {
            if (points.empty()) { return false; }
            
            pathvec_t distances(points.size());
            std::transform(points.begin(), points.end(),
                           distances.begin(),
                           [&](const DataPoint& p) {
                return comparator(*this, p);
            });
            
            path_t tmp;
            int i, j, min_pos, idx,
                im = points.size(),
                Mm = M.size();
            
            // using pair_t = std::pair<path_t, std::size_t>;
            // using vec_t = std::vector<pair_t>;
            // using greater_t = std::greater<typename vec_t::value_type>;
            // using queue_t = std::priority_queue<pair_t, vec_t, greater_t>;
            // queue_t queue;
            // for (i = 0; i < distances.size(); ++i) {
            //     queue.push(pair_t(distances[i], i));
            // }
            // min_pos = queue.top().second;
            
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
    
    public:
        static constexpr std::size_t DL = 1; /// hardcoded
        using DPBase = DataPointBase<Datum<DataType, PathType,
                                           DataLength, PathLength>>;
        using data_t = DataType;
        data_t datum;
        
        struct comparator_t : DPBase::comparator_t {
            PathType operator()(const Datum& d1, const Datum& d2) {
                if (d1.datum == 0 || d2.datum == 0) {
                    return kComparatorBadArguments;
                }
                return (PathType)std::abs((int)d1.datum - (int)d2.datum);
            }
        };
        
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
        void assign(data_t* dptr) {
            datum = dptr[0];
        }
        
        inline DataType* data(int idx = 0) {
            return &datum;
        }
        
        template <typename CastType> inline
        CastType* data_as(int idx = 0) {
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
    
    public:
        static constexpr std::size_t DL = DataLength;
        using DPBase = DataPointBase<Vector<DataType, PathType,
                                            DataLength, PathLength>>;
        using data_t = DataType;
        using ilist_t = std::initializer_list<data_t>;
        using datavec_t = std::vector<data_t>;
        datavec_t datavec;
        
        struct comparator_t : DPBase::comparator_t {
            PathType operator()(const Vector& d1, const Vector& d2) {
                if (d1.datavec.empty() || d2.datavec.empty()) {
                    return kComparatorBadArguments;
                }
                if (d1.datavec.size() != d2.datavec.size()) {
                    return kComparatorUnequalDataVectorLengths;
                }
                unsigned int idx, sum = 0, max = d1.datavec.size();
                for (idx = 0; idx < max; ++idx) {
                    sum += std::abs((int)d1.datavec[idx] - (int)d2.datavec[idx]);
                }
                return (PathType)sum/(PathType)max;
            }
        };
        
        Vector() : DPBase()
            ,datavec(DL)
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
        void assign(data_t* dptr) {
            datavec = datavec_t(dptr);
        }
        
        inline DataType* data(int idx = 0) {
            return &datavec[idx];
        }
        
        template <typename CastType> inline
        CastType* data_as(int idx = 0) {
            return static_cast<CastType*>(data(idx));
        }
        
        template <typename BinaryPredicate> inline
        bool binary_op(const Vector& rhs,
                       BinaryPredicate predicate = BinaryPredicate()) {
            // if (datavec.size() != rhs.datavec.size()) { return false; }
            return std::equal(datavec.begin(),      datavec.end(),
                              rhs.datavec.begin(),  rhs.datavec.end(),
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
        static constexpr std::size_t LCH = LeafCapHigh;
        
        using datapoint_t = DataPoint;
        using datapointer_t = typename DataPoint::DPBase::pointer_t;
        using pointvec_t = typename DataPoint::DPBase::pointvec_t;
        using pathvec_t = std::vector<PathType>;
        using Comparator = typename DataPoint::comparator_t;
        
        enum NodeType : std::size_t {
            ABSTRACT = 0,
            INTERNAL = 1,
            LEAF };
        
        struct InternalTag {};
        struct LeafTag {};
    
    private:
        struct NodeBase {
            NodeType nodetype;
            DataPoint* sv1;
            DataPoint* sv2;
            NodeBase()
                :nodetype(NodeType::ABSTRACT)
                ,sv1(nullptr), sv2(nullptr)
                {}
            inline NodeType type() const { return nodetype; }
            inline bool isInternal() const { return nodetype == NodeType::INTERNAL; }
            inline bool isLeaf() const { return nodetype == NodeType::LEAF; }
            void destroy() { if (sv1 != nullptr) delete sv1;
                             if (sv2 != nullptr) delete sv2; }
            virtual void clear() { destroy(); }
            virtual ~NodeBase() {}
        };
        struct InternalNode : virtual public NodeBase {
            pathvec_t M1, M2;
            std::array<NodeBase*, FanOut> children;
            InternalNode() : NodeBase()
                ,NodeBase::nodetype(NodeType::INTERNAL)
                ,M1(LengthM1), M2(BranchFactor)
                ,children{}
                {}
            ~InternalNode() {}
            void clear(int lvl = 0) override {
                for (auto* node : children) {
                    node->clear(lvl+1);
                }
                NodeBase::destroy();
            }
        };
        struct LeafNode : virtual public NodeBase {
            pathvec_t d1, d2;
            pointvec_t points;
            LeafNode() : NodeBase()
                ,NodeBase::nodetype(NodeType::LEAF)
                ,d1(LeafCap), d2(LeafCap)
                ,points(LeafCap)
                {}
            ~LeafNode() {}
            void clear(int lvl = 0) override {
                NodeBase::destroy();
            }
        };
        struct Node : public InternalNode, public LeafNode {
            explicit Node(InternalTag x)
                :InternalNode()
                {}
            explicit Node(LeafTag x)
                :LeafNode()
                {}
        };
    
    protected:
        using idx_t = std::tuple<std::size_t, std::size_t>;
        using vantagepoints_t = std::tuple<datapointer_t, datapointer_t>;
        int descriptor;
        int knearest;
        Node* top;
        Comparator comparator;
        
        idx_t vantage_indexes(const pointvec_t& points) {
            std::size_t sv1pos = (points.size() >= 1) ? 0 : -1;
            std::size_t sv2pos = -1;
            PathType maxDist = 0.0f, d;
            std::size_t i, j,
                        im = points.size(),
                        jm = points.size();
            for (i = 0; i < im; i++) {
                for (j = i+1; j < jm; j++) {
                    d = comparator(points[i], points[j]);
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
            std::size_t sv1pos, sv2pos;
            std::tie(sv1pos, sv2pos) = vantage_indexes(points);
            return std::make_tuple(points[sv1pos], points[sv2pos]);
        }
        
        Node* addNode(Node* node, pointvec_t& points, int lvl = 0) {
            if (points.empty()) {
                return node;
            }
            
            Node* newNode = node;
            
            if (newNode == NULL || newNode == nullptr) {
                /// create new node
                if (points.size() <= LCH) {
                    /// create leaf node
                    newNode = new Node(LeafTag{});
                    std::tie(newNode->sv1, newNode->sv2) = vantage_points(points);
                    newNode->sv1->distance_range(points, lvl);
                    newNode->sv2->distance_range(points, lvl+1);
                    int i, count = 0;
                    for (i = 0; i < points.size(); i++) {
                        if (*points[i] == *(newNode->sv1)) { continue; }
                        if (*points[i] == *(newNode->sv2)) { continue; }
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
                    /*
                    !!!SORT POINTS!!!
                    !!!SORT POINTS!!!
                    !!!SORT POINTS!!!
                    */
                }
            } else {
                /// node already exists
                if (newNode->isLeaf()) {
                    /// node is a leaf node
                    if (newNode->points.size() + points.size() <= LCp) {
                        /// add points into leaf ("plenty of room")
                    } else {
                        /// not enough room in current leaf --
                        /// create new node
                    }
                } else {
                    /// node is an internal node --
                    /// must recurse on subnodes
                }
            }
            
            /// all is well - reset the tree's root node
            return newNode;
        }
        
    
    public:
        
        Tree()
            :comparator()
            ,descriptor(0)
            ,knearest(0)
            {}
        
        void clear() {}
        
        void add(pointvec_t points) {
            top = addNode(top, points);
        }
        
        // select_vantage_points(); // Tree<T> method(s)
        // find_splits(); // DataPoint method
        // sort_points();
        // 
        // find_distance_range_for_vp(); // DataPoint method!
        //
        // add()
        // retrieve()
        
};

using BasicTree = Tree<uint8_t>;
using VectorTree = Tree<uint8_t, Vector>;

int main() {
    
    BasicTree btree;
    VectorTree vtree;
    
    return 0;
}