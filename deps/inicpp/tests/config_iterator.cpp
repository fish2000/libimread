#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "config.h"
#include "section.h"
#include <vector>

// using namespace inicpp;

TEST(inicpp::config_iterator, construction_and_copying) {
    inicpp::config conf;
    conf.add_section("sect1");
    conf.add_section("sect2");
    conf.add_section("sect3");
    
    // basic constructor
    inicpp::config_iterator<inicpp::section> it(conf, 0);
    inicpp::config_iterator<inicpp::section> it_beg(conf);
    EXPECT_EQ(it, it_beg);
    
    // copy constructor + assignment
    inicpp::config_iterator<inicpp::section> copy_it(it);
    inicpp::config_iterator<inicpp::section> copy_it_assignment = it;
    EXPECT_EQ(copy_it, copy_it_assignment);
    
    // move constructor + assignment
    inicpp::config_iterator<inicpp::section> move_it(std::move(copy_it));
    inicpp::config_iterator<inicpp::section> move_it_assignment = std::move(copy_it_assignment);
    EXPECT_EQ(move_it, move_it_assignment);
}

TEST(inicpp::config_iterator, incrementation) {
    inicpp::config conf;
    conf.add_section("sect1");
    conf.add_section("sect2");
    conf.add_section("sect3");
    inicpp::config_iterator<inicpp::section> it(conf, 0);
    
    EXPECT_EQ(it->get_name(), "sect1");
    EXPECT_EQ((*it).get_name(), "sect1");
    auto postinc_value = it++;
    EXPECT_EQ(postinc_value->get_name(), "sect1");
    EXPECT_EQ(it->get_name(), "sect2");
    auto preinc_value = ++it;
    EXPECT_EQ(preinc_value->get_name(), "sect3");
}

TEST(inicpp::config_iterator, equality_operator) {
    inicpp::config conf;
    conf.add_section("sect1");
    conf.add_section("sect2");
    conf.add_section("sect3");
    inicpp::config_iterator<inicpp::section> it1(conf, 0);
    inicpp::config_iterator<inicpp::section> it2(conf, 0);
    EXPECT_EQ(it1, it1);
    EXPECT_EQ(it1, it2);
    ++it1;
    ++it2;
    EXPECT_EQ(it1, it2);
    ++it2;
    EXPECT_NE(it1, it2);
}
