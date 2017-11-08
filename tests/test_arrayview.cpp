
#define CATCH_CONFIG_FAST_COMPILE

#include <array>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/ext/arrayview.hh>

// #include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    template <std::size_t Rank>
    std::ostream& operator<<(std::ostream& os, av::offset<Rank> const& off) {
        /// av::offset<…> output stream print helper,
        /// from original testsuite
        os << "(" << off[0];
        for (std::size_t i = 1; i < off.rank; ++i) {
            os << "," << off[i];
        }
        return os << ")";
    }
    
    TEST_CASE("[arrayview] Bundled Tests: av::offset initialization",
              "[arrayview][bundled-tests][offset-initialization]")
    {
        av::offset<1> off0;
        CHECK(0 == off0[0]);
        
        av::offset<1> off1(4);
        CHECK(4 == off1[0]);
        
        av::offset<4> off2;
        CHECK(0 == off2[0]);
        CHECK(0 == off2[1]);
        CHECK(0 == off2[2]);
        CHECK(0 == off2[3]);
        
        av::offset<3> off3 = { 1, 2, 3 };
        CHECK(1 == off3[0]);
        CHECK(2 == off3[1]);
        CHECK(3 == off3[2]);
        
    }
    
    TEST_CASE("[arrayview] Bundled Tests: av::bounds size/contains",
              "[arrayview][bundled-tests][bounds-size-contains]")
    {
        av::bounds<3> b = { 2, 3, 4 };
        
        CHECK(24 == b.size()); /// the lone size test
        
        CHECK(b.contains({ 0, 0, 0 }));
        CHECK(b.contains({ 1, 2, 3 }));
        
        CHECK(!b.contains({ 1, 2,  4 }));
        CHECK(!b.contains({ 1, 3,  3 }));
        CHECK(!b.contains({ 2, 2,  3 }));
        CHECK(!b.contains({ 0, 0, -1 }));
        
    }
    
    TEST_CASE("[arrayview] Bundled Tests: av::bounds_iterator begin/end",
              "[arrayview][bundled-tests][bounds_iterator-begin-end]")
    {
        av::bounds<3> b = { 4, 5, 6 };
        av::bounds_iterator<3> iter = b.begin();
        while (iter != b.end()) {
            *iter++;
        }
        CHECK(b.begin() == std::begin(b));
        CHECK(b.end() == std::end(b));
        
    }
    
    TEST_CASE("[arrayview] Bundled Tests: av::bounds_iterator difference",
              "[arrayview][bundled-tests][bounds_iterator-difference]")
    {
        av::bounds<3> b = { 4, 5, 9 };
        av::bounds_iterator<3> iter(b, { 2, 4, 7 });
        iter += 25;
        
        CHECK(3 == (*iter)[0]);
        CHECK(2 == (*iter)[1]);
        CHECK(5 == (*iter)[2]);
        
        iter -= 25;
        CHECK(2 == (*iter)[0]);
        CHECK(4 == (*iter)[1]);
        CHECK(7 == (*iter)[2]);
        
        av::bounds_iterator<3> iter2 = iter + 25;
        CHECK(3 == (*iter2)[0]);
        CHECK(2 == (*iter2)[1]);
        CHECK(5 == (*iter2)[2]);
        
        av::bounds_iterator<3> iter3 = iter2 - 25;
        CHECK(2 == (*iter3)[0]);
        CHECK(4 == (*iter3)[1]);
        CHECK(7 == (*iter3)[2]);
        
        av::offset<3> off = iter3[25];
        CHECK(3 == off[0]);
        CHECK(2 == off[1]);
        CHECK(5 == off[2]);
    }
    
    /// FIXTURES:
    /// FIXTURES:
    /// FIXTURES:
    
    namespace fixies {
    
        class ArrayViewTestFixture {
        
            public:
                ArrayViewTestFixture()
                    :vec(4 * 8 * 12)
                    ,testBounds{ 4, 8, 12 }
                    ,av(vec, testBounds)
                    ,sav(av)
                    {
                        int n{};
                        std::generate(vec.begin(),
                                      vec.end(),
                                      [&]{ return n++; });
                        testStride = av.stride();
                    }
            
            protected:
                std::vector<int> vec;
                av::bounds<3> testBounds;
                av::offset<3> testStride;
                av::array_view<int, 3> av;              /// objects under test, one of each
                av::strided_array_view<int, 3> sav;     /// …but both under contigious data
        
        }; /// class ArrayViewTestFixture
        
        using StridedArrayViewTestFixture = ArrayViewTestFixture;
        
        class StridedDataTestFixture : public ArrayViewTestFixture {
            
            public:
                using Avt = ArrayViewTestFixture;
            
            public:
                StridedDataTestFixture()
                    :testBounds{ Avt::testBounds[0],
                                 Avt::testBounds[1],
                                 Avt::testBounds[2] / 2 }
                    ,testStride{ Avt::testStride[0],
                                 Avt::testStride[1],
                                 2 }
                    ,strided_sav(vec.data(), testBounds, testStride)
                    {}
            
            protected:
                av::bounds<3> testBounds;
                av::offset<3> testStride;
                av::strided_array_view<int, 3> strided_sav;     /// object under test, a strided class
                                                                /// with alternate spacing in z
                                                                /// … even's only!
        }; /// class StridedDataTestFixture
    
    } /// namespace fixies
    
    /// TEST FUNCTION TEMPLATES:
    /// TEST FUNCTION TEMPLATES:
    /// TEST FUNCTION TEMPLATES:
    
    namespace tmpl {
    
        template <typename ArrayView>
        void subsectioning(ArrayView const& av, av::bounds<3> const& testBounds,
                                                av::offset<3> const& testOrigin,
                                                av::offset<3> const& testStride) {
            int start{};
            
            for (int ix = 0; ix < 3; ix++) {
                start += testOrigin[ix] * testStride[ix];
            }
            
            av::bounds_iterator<3> iter = std::begin(av.bounds());
            for (int ii = 0; ii < testBounds[0]; ++ii) {
                for (int jj = 0; jj < testBounds[1]; ++jj) {
                    for (int kk = 0; kk < testBounds[2]; ++kk) {
                        av::offset<3> idx = { ii, jj, kk };
                        CHECK(idx == *iter);
                        int off{};
                        for (int d = 0; d < 3; d++) {
                            off += idx[d] * testStride[d];
                        }
                        CHECK((start + off) == av[*iter++]);
                    }
                }
            }
        }
        
        template <typename ArrayView>
        void dimensionslicing(ArrayView const& av, av::offset<3> const& testStride) {
            // Slices always fix the most significant dimension
            int x = 2;
            av::strided_array_view<int, 2> sliced = av[x];
            int start = testStride[0] * x;
            for (av::bounds_iterator<2> iter = std::begin(sliced.bounds());
                                        iter != std::end(sliced.bounds());
                                      ++iter) {
                CHECK(start == sliced[*iter]);
                start += testStride[2];
            }
            
            // Cascade slices
            int y = 3;
            av::strided_array_view<int, 1> sliced2 = av[x][y];
            int start2 = testStride[0] * x + testStride[1] * y;
            for (av::bounds_iterator<1> iter = std::begin(sliced2.bounds());
                                        iter != std::end(sliced2.bounds());
                                      ++iter) {
                CHECK(start2 == sliced2[*iter]);
                start2 += testStride[2];
            }
            
            // Cascade to a single index
            int z = 3;
            int start3 = testStride[0] * x + testStride[1] * y + testStride[2] * z;
            CHECK(start3 == av[x][y][z]);
        }
    
    } /// namespace tmpl
    
    TEST_CASE_METHOD(fixies::ArrayViewTestFixture,
                    "[arrayview] Bundled Tests: ArrayViewTest and constructors",
                    "[arrayview][bundled-tests][fixtures][ArrayViewTest-constructors]")
    {
        int start{};
        for (av::bounds_iterator<3> iter = std::begin(av.bounds());
                                    iter != std::end(av.bounds());
                                  ++iter) {
            CHECK(start++ == av[*iter]);
        }
    }
    
    TEST_CASE_METHOD(fixies::ArrayViewTestFixture,
                    "[arrayview] Bundled Tests: ArrayViewTest and observers",
                    "[arrayview][bundled-tests][fixtures][ArrayViewTest-observers]")
    {
        CHECK(av.bounds() == testBounds);
        CHECK(av.size() == (testBounds[0] * testBounds[1] * testBounds[2]));
        CHECK(av.stride() == testStride);
    }
    
    TEST_CASE_METHOD(fixies::ArrayViewTestFixture,
                    "[arrayview] Bundled Tests: ArrayViewTest and slicing",
                    "[arrayview][bundled-tests][fixtures][ArrayViewTest-slicing]")
    {
        tmpl::dimensionslicing(av, testStride);
    }
    
    TEST_CASE_METHOD(fixies::ArrayViewTestFixture,
                    "[arrayview] Bundled Tests: ArrayViewTest and sectioning",
                    "[arrayview][bundled-tests][fixtures][ArrayViewTest-sectioning]")
    {
        av::offset<3> origin{ 1, 2, 3 };
        
        // section with new bounds
        av::bounds<3> newBounds{ 2, 3, 4 };
        tmpl::subsectioning(av.section(origin, newBounds),
                                       newBounds,
                                       origin,
                                       testStride);
        
        // section with bounds extending to extent of source view
        av::strided_array_view<int, 3> sectioned = av.section(origin);
        
        av::bounds<3> remainingBounds = testBounds - origin;
        CHECK(remainingBounds == sectioned.bounds());
        tmpl::subsectioning(sectioned, remainingBounds,
                                       origin,
                                       testStride);
    }
    
    TEST_CASE_METHOD(fixies::StridedArrayViewTestFixture,
                    "[arrayview] Bundled Tests: StridedArrayViewTest and constructors",
                    "[arrayview][bundled-tests][fixtures][StridedArrayViewTest-constructors]")
    {
        // Default
        av::strided_array_view<int, 3> sav{};
        CHECK(0 == sav.size());
        
        // From array_view
        av::strided_array_view<int, 3> sav2(av);
        int iix{};
        for (av::bounds_iterator<3> iter = std::begin(sav2.bounds());
                                    iter != std::end(sav2.bounds());
                                  ++iter) {
            CHECK(iix++ == sav2[*iter]);
        }
        
        // From strided_array_view
        av::strided_array_view<int, 3> sav3(sav2);
        int iiz{};
        for (av::bounds_iterator<3> iter = std::begin(sav3.bounds());
                                    iter != std::end(sav3.bounds());
                                  ++iter) {
            CHECK(iiz++ == sav3[*iter]);
        }
    }
    
    TEST_CASE_METHOD(fixies::StridedArrayViewTestFixture,
                    "[arrayview] Bundled Tests: StridedArrayViewTest and observers",
                    "[arrayview][bundled-tests][fixtures][StridedArrayViewTest-observers]")
    {
        CHECK(sav.bounds() == testBounds);
        CHECK(sav.size() == (testBounds[0] * testBounds[1] * testBounds[2]));
        CHECK(sav.stride() == testStride);
    }
    
    TEST_CASE_METHOD(fixies::StridedArrayViewTestFixture,
                    "[arrayview] Bundled Tests: StridedArrayViewTest and slicing",
                    "[arrayview][bundled-tests][fixtures][StridedArrayViewTest-slicing]")
    {
        tmpl::dimensionslicing(sav, testStride);
    }
    
    TEST_CASE_METHOD(fixies::StridedArrayViewTestFixture,
                    "[arrayview] Bundled Tests: StridedArrayViewTest and sectioning",
                    "[arrayview][bundled-tests][fixtures][StridedArrayViewTest-sectioning]")
    {
        av::offset<3> origin{ 1, 2, 3 };
        
        // section with new bounds
        av::bounds<3> newBounds{ 2, 3, 4 };
        tmpl::subsectioning(sav.section(origin, newBounds),
                                                newBounds,
                                                origin,
                                                testStride);
        
        // section with bounds extending to extent of source view
        av::strided_array_view<int, 3> sectioned = sav.section(origin);
        
        av::bounds<3> remainingBounds = testBounds - origin;
        CHECK(remainingBounds == sectioned.bounds());
        tmpl::subsectioning(sectioned, remainingBounds,
                                       origin,
                                       testStride);
    }
    
    TEST_CASE_METHOD(fixies::StridedDataTestFixture,
                    "[arrayview] Bundled Tests: StridedDataTest and constructors",
                    "[arrayview][bundled-tests][fixtures][StridedDataTest-constructors]")
    {
        int start{};
        for (av::bounds_iterator<3> iter = std::begin(strided_sav.bounds());
                                    iter != std::end(strided_sav.bounds());
                                  ++iter) {
            CHECK(start++ == strided_sav[*iter]);
            start++;
            // start += 2; /// <---- enabling this (from the original testsuite)
                           /// -- instead of the second post-autoincrement -- causes
                           /// a rash of failures to issue from the preceding CHECK()
        }
    }
    
    TEST_CASE_METHOD(fixies::StridedDataTestFixture,
                    "[arrayview] Bundled Tests: StridedDataTest and observers",
                    "[arrayview][bundled-tests][fixtures][StridedDataTest-observers]")
    {
        CHECK(strided_sav.bounds() == testBounds);
        CHECK(strided_sav.size() == (testBounds[0] * testBounds[1] * testBounds[2]));
        CHECK(strided_sav.stride() == testStride);
    }
    
    TEST_CASE_METHOD(fixies::StridedDataTestFixture,
                    "[arrayview] Bundled Tests: StridedDataTest and slicing",
                    "[arrayview][bundled-tests][fixtures][StridedDataTest-slicing]")
    {
        tmpl::dimensionslicing(strided_sav, testStride);
    }
    
    TEST_CASE_METHOD(fixies::StridedDataTestFixture,
                    "[arrayview] Bundled Tests: StridedDataTest and sectioning",
                    "[arrayview][bundled-tests][fixtures][StridedDataTest-sectioning]")
    {
        av::offset<3> origin{ 1, 2, 3 };
        
        // section with new bounds
        av::bounds<3> newBounds{ 2, 3, 4 };
        tmpl::subsectioning(strided_sav.section(origin, newBounds),
                                                        newBounds,
                                                        origin,
                                                        testStride);
        
        // section with bounds extending to extent of source view
        av::strided_array_view<int, 3> sectioned = strided_sav.section(origin);
        
        av::bounds<3> remainingBounds = testBounds - origin;
        CHECK(remainingBounds == sectioned.bounds());
        tmpl::subsectioning(sectioned, remainingBounds,
                                       origin,
                                       testStride);
    }
    
    TEST_CASE("[arrayview] Bundled Tests: programming example",
              "[arrayview][bundled-tests][programming-example]")
    {
        int X = 12;
        int Y = 8;
        int Z = 6;
        
        std::vector<int> vec(X * Y * Z);
        av::bounds<3> extents = { X, Y, Z };
        av::array_view<int, 3> av(vec, extents);
        
        // Access an element for writing:
        av::offset<3> idx = { 5, 3, 2 };
        av[idx] = 30;
        
        // Iterate through each index of the view…
        for (auto& idx : av.bounds()) {
            auto i = idx[0];
            auto j = idx[1];
            auto k = idx[2];
            av[idx] = i * j * k;
        }
        
        // …or use a bounds_iterator explicitly:
        av::bounds_iterator<3> first = std::begin(av.bounds());
        av::bounds_iterator<3> last = std::end(av.bounds());
        
        std::for_each(first, last, [&](av::offset<3> const& idx) {
            auto i = idx[0];
            auto j = idx[1];
            auto k = idx[2];
            // std::cout << i << " " << j << " " << k << std::endl;
            CHECK((i * j * k) == av[idx]);
        });
        
        // Slicing
        int x0 = 5;
        int y0 = 3;
        av::array_view<int, 2> slice2d = av[x0];      // a 2d slice in the yz plane
        av::array_view<int, 1> slice1d = av[x0][y0];  // a row in z (also the contigious dimension)
        
        CHECK(30 == slice2d[{ 3, 2 }]);
        CHECK(30 == slice1d[2]);
        
        // Sectioning
        av::offset<3> origin = { 6, 3, 2 };
        av::bounds<3> window = { 3, 3, 2 };
        auto section = av.section(origin, window);
        
        int sum = std::accumulate(std::begin(section.bounds()),
                                  std::end(section.bounds()),
                                  0, [&](int a, av::offset<3> idx) { return a + section[idx]; });
        
        // std::cout << std::endl << "SUM = " << sum
        //           << std::endl
        //           << std::endl;
        
        CHECK(1260 == sum);
        
        // Strided data
        av::offset<3> newStride = { av.stride()[0],
                                    av.stride()[1],
                                    3 };
        
        av::bounds<3> newExtents = { X, Y, Z/3 };
        
        av::strided_array_view<int, 3> sav(vec.data(), newExtents, newStride);
        
        for (auto& idx : sav.bounds()) { CHECK(0 == (sav[idx] % 3)); }
    }
} /// namespace (anon.)