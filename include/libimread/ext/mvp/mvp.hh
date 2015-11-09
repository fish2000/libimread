#include <cstdlib>
#include <cmath>

#include <array>
#include <deque>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <initializer_list>

static const float kComparatorBadArguments = -1.0f;
static const float kComparatorUnequalDataVectorLengths = -2.0f;

template <typename DataType,
          typename PathType = float,
          std::size_t DataLength = 1, /// dummy parameter
          std::size_t PathLength = 5>
class Datum {
    
    public:
        static constexpr std::size_t DL = 1; /// hardcoded
        static constexpr std::size_t PL = PathLength;
        static constexpr std::size_t DS = sizeof(DataType);
        
        using data_t = DataType;
        using path_t = PathType;
        using plist_t = std::initializer_list<path_t>;
        using datavec_t = std::vector<data_t>;
        using pathlist_t = std::array<path_t, PL>;
        
        using comparator_t = std::function<path_t(const Datum&,
                                                  const Datum&)>;
        
        std::string name;
        data_t datum;
        pathlist_t pathlist;
        
        comparator_t comparator = [&](const Datum& d1,
                                      const Datum& d2) {
            if (d1.datum == 0 || d2.datum == 0) {
                return kComparatorBadArguments;
            }
            return (path_t)std::abs((int)d1.datum - (int)d2.datum);
        };
        
        Datum()
            :name("")
            ,datum(0)
            ,pathlist{ 0 }
            {}
        
        explicit Datum(data_t d)
            :name("")
            ,datum(d)
            ,pathlist{ 0 }
            {}
        
        Datum(const Datum& other)
            :name(other.name)
            ,datum(other.datum)
            ,pathlist(other.pathlist)
            {}
        
        Datum(Datum&& other)
            :name(std::move(other.name))
            ,datum(std::move(other.datum))
            ,pathlist(std::move(other.pathlist))
            {}
        
        void set_paths(plist_t plist) {
            pathlist = pathlist_t(plist);
        }
        
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
        
        inline PathType compare(const Datum& other) {
            return comparator(*this, other);
        }
        
};

template <typename DataType,
          typename PathType = float,
          std::size_t DataLength = 10, /// this is arbitrary
          std::size_t PathLength = 5>
class Vector {
    
    public:
        static constexpr std::size_t DL = DataLength;
        static constexpr std::size_t PL = PathLength;
        static constexpr std::size_t DS = sizeof(DataType);
        
        using data_t = DataType;
        using path_t = PathType;
        using ilist_t = std::initializer_list<data_t>;
        using plist_t = std::initializer_list<path_t>;
        using datavec_t = std::vector<data_t>;
        using pathlist_t = std::array<path_t, PL>;
        using comparator_t = std::function<path_t(const Vector&,
                                                  const Vector&)>;
        
        std::string name;
        datavec_t datavec;
        pathlist_t pathlist;
        
        comparator_t comparator = [&](const Vector& d1,
                                      const Vector& d2) {
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
            return (path_t)sum/(path_t)max;
        };
        
        Vector()
            :name("")
            ,datavec(DL)
            ,pathlist{ 0 }
            {}
        
        explicit Vector(ilist_t ilist)
            :name("")
            ,datavec(ilist)
            ,pathlist{ 0 }
            {}
        
        Vector(const Vector& other)
            :name(other.name)
            ,datavec(other.datavec)
            ,pathlist(other.pathlist)
            {}
        
        Vector(Vector&& other)
            :name(std::move(other.name))
            ,datavec(std::move(other.datavec))
            ,pathlist(std::move(other.pathlist))
            {}
        
        void set_paths(plist_t plist) {
            pathlist = pathlist_t(plist);
        }
        
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
        
        inline PathType compare(const Vector& other) {
            return comparator(*this, other);
        }
        
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
        using datavec_t = std::vector<DataPoint>;
        using Comparator = typename DataPoint::comparator_t;
        
        enum NodeType : std::size_t {
            ABSTRACT = 0,
            INTERNAL = 1,
            LEAF
        };
    
    private:
        struct Node {
            virtual NodeType type() const { return NodeType::ABSTRACT; }
            DataPoint* sv1;
            DataPoint* sv2;
            Node()
                :sv1(NULL), sv2(NULL) {}
            ~Node() { if (sv1 != NULL) delete sv1;
                      if (sv2 != NULL) delete sv2; }
        };
        struct InternalNode : public Node {
            NodeType type() const override { return NodeType::INTERNAL; }
            std::vector<PathType> M1;
            std::vector<PathType> M2;
            std::array<Node*, FanOut> children;
            InternalNode() :Node()
                ,children{}
                ,M1(LengthM1), M2(BranchFactor)
                {}
        };
        struct LeafNode : public Node {
            NodeType type() const override { return NodeType::LEAF; }
            std::vector<PathType> d1;
            std::vector<PathType> d2;
            std::deque<DataPoint*> points;
            LeafNode() :Node()
                ,points(LeafCap)
                ,d1(LeafCap), d2(LeafCap)
                {}
        };
    
    protected:
        int descriptor;
        int knearest;
        Node* top;
        Comparator comparator;
    
    public:
        
        Tree()
            :comparator(DataPoint::comparator)
            ,descriptor(0)
            ,knearest(0)
            {}
        
        // template <typename T>
        // void add(T&& t) {
        //  DataPoint(std::forward<T>(t))
        // }
        
        
        
};

using BasicTree = Tree<uint8_t>;
using VectorTree = Tree<uint8_t, Vector>;

int main() {
    
    BasicTree btree();
    VectorTree vtree();
    
    return 0;
}