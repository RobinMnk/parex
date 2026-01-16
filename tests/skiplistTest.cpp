//
// Created by robin on 07.05.25.
//

#include "../lib/utils/skiplist.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_set>

class SkipListTest : public ::testing::TestWithParam<size_t> {
protected:
    static std::mt19937 gen;
    static std::random_device rd;

    template<typename T>
    static void randomPermutation(std::vector<T>& v) {
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), gen);
    }

    static int randInt(size_t max) {
        std::uniform_int_distribution<> dis(0, max-1);
        return dis(gen);
    }
};

std::random_device SkipListTest::rd{};
std::mt19937 SkipListTest::gen{23121};

TEST_F(SkipListTest, ConstructEmpty) {
    SkipList sl{0};
}

TEST_P(SkipListTest, Construct) {
    size_t length = GetParam();
    SkipList sl{length};
}

TEST_P(SkipListTest, SingleDelete) {
    size_t length = GetParam();
    SkipList sl{length};
    sl.del(length/2);
}

void checkEqual(const SkipList& sl, const std::unordered_set<size_t>& groundTruth, size_t n) {
    std::unordered_set<size_t> elements;
    for(size_t i = sl.begin(); i < n; i += sl.get(i)) {
        if(!groundTruth.contains(i)) {
            printf("here\n");
        }
        ASSERT_TRUE(groundTruth.contains(i)) << i << " in skiplist but not in groundTruth";
        elements.insert(i);
    }

    for(size_t i: groundTruth) {
        if(!elements.contains(i)) {
            printf("here\n");
        }
        ASSERT_TRUE(elements.contains(i)) << i << " in groundTruth but not in skiplist";
    }
}

TEST_P(SkipListTest, DeleteAllRandom) {
    size_t length = GetParam();
    SkipList sl{length};

    std::vector<size_t> deletions(length);
    std::iota(deletions.begin(), deletions.end(), 0);
    randomPermutation(deletions);

    std::unordered_set<size_t> groundTruth(deletions.begin(), deletions.end());

    for(size_t i = 0; i < length; i++) {
        sl.del(deletions[i]);
        groundTruth.erase(deletions[i]);
        checkEqual(sl, groundTruth, length);
    }
}


TEST_P(SkipListTest, DeleteAllRandomRange) {
    size_t length = GetParam() * 100;
    SkipList sl{length};

    std::vector<size_t> deletions(length);
    std::iota(deletions.begin(), deletions.end(), 0);
    randomPermutation(deletions);

    std::unordered_set<size_t> groundTruth(deletions.begin(), deletions.end());

    for (size_t i = 0; i < length; ++i) {
        size_t current = deletions[i];
        if (current >= length || !groundTruth.contains(current)) continue;

        int extraDeletions = groundTruth.size() > 4 ? randInt(groundTruth.size() / 4) : 0;
        int actualDeletions = 0;

        // Only delete items within bounds and present in the set
        for (int k = 1; k <= extraDeletions; ++k) {
            size_t candidate = current + k;
            if (candidate >= length || !groundTruth.contains(candidate)) break;
            ++actualDeletions;
        }

        // Call deletion on the skip list
        sl.del(current, actualDeletions);

        // Erase from ground truth
        groundTruth.erase(current);
        for (int j = 1; j <= actualDeletions; ++j) {
            groundTruth.erase(current + j);
        }

        // Verify correctness
        checkEqual(sl, groundTruth, length);
    }
}

INSTANTIATE_TEST_SUITE_P(
    InstructionSequences,
    SkipListTest,
    testing::Values(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
);