//
// Created by robin on 07.05.25.
//

#ifndef RCUT_SKIPLIST_H
#define RCUT_SKIPLIST_H

#include <cstddef>
#include <vector>

class SkipList {
    size_t start{0};
    std::vector<int> skip;

public:
    explicit SkipList(size_t length) : skip(length, 1) {}

    void del(size_t index, size_t elems=0) {
//        assert(skip[index] > 0);
        if(index == start) {
            start += skip[index + elems] + elems;
            return;
        }

        size_t pred = index - 1;
        if(skip[pred] < 0) { // pred is deleted
            pred += skip[pred];
        }
        size_t step = skip[index + elems] + elems;
        size_t succ = index + step;

        skip[pred] += step;
        skip[succ-1] = pred - succ + 1;
    }

    [[nodiscard]] inline int get(size_t index) const {
        return skip[index];
    }

    [[nodiscard]] inline size_t begin() const {
        return start;
    }
};


#endif //RCUT_SKIPLIST_H
