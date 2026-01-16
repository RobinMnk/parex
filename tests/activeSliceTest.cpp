//
// Created by robin on 20.03.25.
//

#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include "../lib/utils/active_slice.h"

enum Action {
    INSERT_ACTIVE = 0, INSERT_INACTIVE, DEACTIVATE, NOP
};

struct Instruction {
    Action action;
    size_t value;
};

using InstructionList = std::vector<Instruction>;

class ActiveSliceTest : public ::testing::TestWithParam<size_t> {
protected:
    static std::mt19937 gen;
    static std::uniform_int_distribution<> valueDis;
    static std::uniform_int_distribution<> actionDis;
    static std::uniform_int_distribution<> actionDisWithNop;

    static std::vector<size_t> randomList(size_t n) {
        std::vector<size_t> numbers;
        for (size_t i = 0; i < n; ++i) {
            numbers.push_back(i);
        }
        return numbers;
    }


    template <typename T>
    static T randElement(const std::unordered_multiset<T>& set) {
        if (set.empty()) {
            throw std::out_of_range("The unordered_multiset is empty");
        }

        std::uniform_int_distribution<> dis(0, set.size() - 1);  // Random index distribution

        auto it = set.begin();
        std::advance(it, dis(gen));  // Advance the iterator to the random position

        return *it;  // Return the element at the random position
    }

    static int randInt() {
        return valueDis(gen);
    }

    static InstructionList buildInstructions(size_t n) {
        InstructionList instr(n);
        std::unordered_multiset<size_t> active{};
        size_t val;

        for (size_t i = 0; i < n; ++i) {
            auto action = static_cast<Action>(actionDis(gen));
            if(active.empty() && action == DEACTIVATE) continue;

            switch (action) {
                case NOP: break;
                case INSERT_ACTIVE:
                    val = i;
                    active.insert(val);
                    instr[i] = {INSERT_ACTIVE, val};
                    break;
                case INSERT_INACTIVE:
                    instr[i] = {INSERT_INACTIVE, i};
                    break;
                case DEACTIVATE:
                    size_t elem = randElement(active);
                    active.erase(elem);
                    instr[i] = {DEACTIVATE, elem};
                    break;
            }
        }

        return instr;
    }

};

std::mt19937 ActiveSliceTest::gen{23121};
std::uniform_int_distribution<> ActiveSliceTest:: valueDis(1, 100);
std::uniform_int_distribution<> ActiveSliceTest:: actionDis(0, 2);
std::uniform_int_distribution<> ActiveSliceTest:: actionDisWithNop(0, 3);


template <typename T, template <typename...> class Container>
std::unordered_multiset<T> getActiveElements(const ActiveSlice<T, Container>& slice) {
    std::unordered_multiset<T> activeElements;
    for(T& elem: slice) {
        activeElements.insert(elem);
    }
    return activeElements;
}

TEST_F(ActiveSliceTest, Construct) {
    std::vector<size_t> list = randomList(10);
    auto activeSlice = buildActiveSlice(list);
    activeSlice.activateAll();
    auto activeElements = getActiveElements(activeSlice);
    ASSERT_EQ(activeElements.size(), 10);
}

TEST_F(ActiveSliceTest, Insert) {
    std::vector<size_t> list = randomList(10);
    auto activeSlice = buildActiveSlice(list);
    activeSlice.activateAll();
    activeSlice.insert(3000, true);
    activeSlice.insert(5000, false);
    auto activeElements = getActiveElements(activeSlice);
    ASSERT_EQ(activeElements.size(), 11);
    ASSERT_TRUE(activeElements.contains(3000));
    ASSERT_TRUE(!activeElements.contains(5000));
    activeSlice.insert(7000, true);
    activeElements = getActiveElements(activeSlice);
    ASSERT_TRUE(activeElements.contains(7000));
}

template <typename T, template <typename...> class Container>
void checkCorrectness(const ActiveSlice<T, Container>& activeSlice, std::unordered_multiset<size_t>& ground_truth) {
    ASSERT_EQ(activeSlice.numActive(), ground_truth.size());

    for(T& elem: activeSlice) {
        ASSERT_TRUE(ground_truth.contains(elem));
    }

    auto activeElements = getActiveElements(activeSlice);

    for(const T& elem: ground_truth) {
        ASSERT_TRUE(activeElements.contains(elem));
    }
}

TEST_P(ActiveSliceTest, Async) {
    size_t numInstr = GetParam();
    InstructionList instrList = buildInstructions(numInstr);

    std::unordered_multiset<size_t> ground_truth{};
    std::vector<size_t> cont{};
    auto activeSlice = buildActiveSlice(cont);

    for(Instruction instr: instrList) {
        switch (instr.action) {
            case NOP:
                break;
            case INSERT_ACTIVE:
//                std::cout << "INSERT_ACTIVE(" << instr.value << "), ";
                ground_truth.insert(instr.value);
                activeSlice.insert(std::move(instr.value), true);
                break;
            case INSERT_INACTIVE:
//                std::cout << "INSERT_INACTIVE(" << instr.value << "), ";
                activeSlice.insert(std::move(instr.value), false);
                break;
            case DEACTIVATE:
//                std::cout << "DEACTIVATE(" << instr.value << "), ";
                ground_truth.erase(instr.value);
                auto it = std::find_if(activeSlice.begin(), activeSlice.end(), [&instr](auto x){return x == instr.value;});
                ASSERT_NE(it, activeSlice.end()) << "Element should have been found!";
                activeSlice.deactivate(it);
                break;
        }
        checkCorrectness(activeSlice, ground_truth);
    }
    std::cout << std::endl;
}

TEST_P(ActiveSliceTest, Concurrent) {
    size_t numElems = GetParam();
    std::vector<size_t> list = randomList(numElems);
    std::unordered_multiset<size_t> ground_truth{};
    std::unordered_multiset<size_t> unprocessed{list.begin(), list.end()};
    ground_truth.insert(list.begin(), list.end());
    auto activeSlice = buildActiveSlice(list);
    activeSlice.activateAll();

    auto it = activeSlice.begin();
    size_t id{0};
    while(it != activeSlice.end()) {
        auto action = static_cast<Action>(actionDisWithNop(gen));

        size_t iterIx = std::distance(activeSlice.begin(), it);
        ASSERT_LE(iterIx, ground_truth.size());

        auto hitp = unprocessed.find(*it);
        ASSERT_TRUE(hitp != unprocessed.end()) << "Processing unknown element: " << *it;
        unprocessed.erase(hitp);

//        std::cout << " - processing: " << *it << std::endl;

        if(action == INSERT_ACTIVE) {
//            std::cout << "INSERT_ACTIVE(" << id << ")" << std::endl;
            ground_truth.insert(id);
            size_t movedId = id;
            size_t before = *it;
            activeSlice.insert(std::move(movedId), it, true);
            ASSERT_EQ(before, *it);
            checkCorrectness(activeSlice, ground_truth);
            ++it;
        } else if (action == INSERT_INACTIVE) {
//            std::cout << "INSERT_INACTIVE(" << id << ")" << std::endl;
            size_t movedId = id;
            size_t before = *it;
            activeSlice.insert(std::move(movedId), it, false);
            ASSERT_EQ(before, *it);
            checkCorrectness(activeSlice, ground_truth);
            ++it;
        } else if (action == DEACTIVATE) {
//            std::cout << "DEACTIVATE(" << *it << ")" << std::endl;
            auto hit(ground_truth.find(*it));
            if (hit != ground_truth.end()) ground_truth.erase(hit);
            size_t before = *it;
            auto nit = activeSlice.deactivate(it);
            ASSERT_EQ(before, *it);
            it = nit;
            checkCorrectness(activeSlice, ground_truth);
        } else if (action == NOP) {
            ++it;
        }
        ++id;
    }

    ASSERT_TRUE(unprocessed.empty());
}


INSTANTIATE_TEST_SUITE_P(
        InstructionSequences,
        ActiveSliceTest,
        testing::Values(2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
);