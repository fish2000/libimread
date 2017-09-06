#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "option.h"
#include "section.h"
#include <vector>

// using namespace inicpp;
using namespace std::literals;

TEST(inicpp::section_iterator, construction_and_copying) {
    inicpp::section sec("section name");
    sec.add_option("opt1", "value1");
    sec.add_option("opt2", "value2");
    sec.add_option("opt3", "value3");
    
    // basic constructor
    inicpp::section_iterator<inicpp::option> it(sec, 0);
    inicpp::section_iterator<inicpp::option> it_beg(sec);
    EXPECT_EQ(it, it_beg);
    
    // copy constructor + assignment
    inicpp::section_iterator<inicpp::option> copy_it(it);
    inicpp::section_iterator<inicpp::option> copy_it_assignment = it;
    EXPECT_EQ(copy_it, copy_it_assignment);
    
    // move constructor + assignment
    inicpp::section_iterator<inicpp::option> move_it(std::move(copy_it));
    inicpp::section_iterator<inicpp::option> move_it_assignment = std::move(copy_it_assignment);
    EXPECT_EQ(move_it, move_it_assignment);
}

TEST(inicpp::section_iterator, incrementation) {
    inicpp::section sec("section name");
    sec.add_option("opt1", "value1");
    sec.add_option("opt2", "value2");
    sec.add_option("opt3", "value3");
    inicpp::section_iterator<inicpp::option> it(sec, 0);
    
    EXPECT_EQ(it->get_name(), "opt1");
    EXPECT_EQ((*it).get_name(), "opt1");
    auto postinc_value = it++;
    EXPECT_EQ(postinc_value->get_name(), "opt1");
    EXPECT_EQ(it->get_name(), "opt2");
    auto preinc_value = ++it;
    EXPECT_EQ(preinc_value->get_name(), "opt3");
}

TEST(inicpp::section_iterator, equality_operator) {
    inicpp::section sec("section name");
    sec.add_option("opt1", "value1");
    sec.add_option("opt2", "value2");
    sec.add_option("opt3", "value3");
    inicpp::section_iterator<inicpp::option> it1(sec, 0);
    inicpp::section_iterator<inicpp::option> it2(sec, 0);
    EXPECT_EQ(it1, it1);
    EXPECT_EQ(it1, it2);
    ++it1;
    ++it2;
    EXPECT_EQ(it1, it2);
    ++it2;
    EXPECT_NE(it1, it2);
}
