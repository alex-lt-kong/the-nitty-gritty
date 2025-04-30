#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <print>

struct Resource {
    int value;
};

TEST(SharedPtrDemo, RefCountGoesUpAndDown) {
    auto shared0 = std::make_shared<Resource>();
    shared0->value = 100;
    EXPECT_EQ(shared0->value, 100);
    EXPECT_EQ(shared0.use_count(), 1);

    auto shared1 = shared0;
    EXPECT_EQ(shared0->value, shared1->value);
    EXPECT_EQ(shared0.use_count(), 2);

    EXPECT_EQ(shared0, shared1);
    // The address-of operator (ampersand) gets the address of the shared_ptr itself,
    // not the address of the hidden pointer (formally known as a "stored pointer") that actually points to the resource
    EXPECT_NE(&shared0, &shared1);
    // get() return the value of the stored pointer
    EXPECT_EQ(shared0.get(), shared1.get());

    const auto shared2 = shared0;
    EXPECT_EQ(shared2->value, shared1->value);
    EXPECT_EQ(shared0.use_count(), 3);
    EXPECT_EQ(shared0.use_count(), shared1.use_count());
    EXPECT_EQ(shared1.use_count(), shared2.use_count());
    //
    {
        auto shared3 = shared0;
        EXPECT_EQ(shared3, shared1);
        EXPECT_EQ(shared0.use_count(), 4);
        EXPECT_EQ(shared0.use_count(), shared1.use_count());
        EXPECT_EQ(shared1.use_count(), shared2.use_count());
        EXPECT_EQ(shared2.use_count(), shared3.use_count());
    }
    EXPECT_EQ(shared0.use_count(), 3);
    EXPECT_EQ(shared0.use_count(), shared1.use_count());
    EXPECT_EQ(shared1.use_count(), shared2.use_count());


    EXPECT_EQ(shared2->value, shared1->value);

    // reset() unlinks the shared_ptr to the underlying object:
    // https://www.reddit.com/r/cpp_questions/comments/v37d0w/understanding_shared_ptrreset/
    shared0.reset();
    EXPECT_EQ(shared0.get(), nullptr);
    EXPECT_EQ(shared0.use_count(), 0);

    EXPECT_EQ(shared1.use_count(), 2);
    EXPECT_NE(shared2.get(), nullptr);
    EXPECT_EQ(shared2->value, 100);
}

TEST(SharedPtrDemo, ReferenceCycle) {
    struct B;

    struct A {
        int *m_dtor_call_count;
        ~A() { ++(*m_dtor_call_count); }
        explicit A(int *dtor_call_count) { m_dtor_call_count = dtor_call_count; }
        std::shared_ptr<B> b;
    };

    struct B {
        int *m_dtor_call_count;
        ~B() { ++(*m_dtor_call_count); }
        explicit B(int *dtor_call_count) { m_dtor_call_count = dtor_call_count; }
        std::shared_ptr<A> a;
    };
    int a_destructed_count = 0;
    int b_destructed_count = 0;

    EXPECT_EQ(a_destructed_count, 0); {
        auto ptrA = std::make_shared<A>(&a_destructed_count);
    }
    EXPECT_EQ(a_destructed_count, 1);
    EXPECT_EQ(b_destructed_count, 0); {
        const auto ptr_a = std::make_shared<A>(&a_destructed_count);
        const auto ptr_b = std::make_shared<B>(&b_destructed_count);
        ptr_a->b = ptr_b;
        ptr_b->a = ptr_a;
    }
    // Funny, still 0
    EXPECT_EQ(b_destructed_count, 0);
}
