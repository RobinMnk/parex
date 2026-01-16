//
// Created by robin on 18.03.25.
//

#ifndef RCUT_ACTIVE_SLICE_H
#define RCUT_ACTIVE_SLICE_H

#include <vector>
#include <numeric> // iota


/**
 * Maintains an active subset of the given Container, subject to insertions and deletions _concurrent_ to iterating active elements.
 * @tparam T
 * @tparam Container
 */
template <typename T, template <typename...> class Container>
class ActiveSlice {
    using ContainerType = Container<T>;

    ContainerType* m_cont;
    std::vector<size_t> indices;
    size_t activeIndex;

    class ActiveSliceIterator {
        T* elem;
        size_t iterationIndex;
        ContainerType* container;
        const std::vector<size_t>* iterIndices;

        void setPointer(size_t index) {
            size_t indexInCont = (*iterIndices)[index];
            elem = &((*container)[indexInCont]);
        }

        friend ActiveSlice<T, Container>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = ContainerType::iterator::difference_type;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        explicit ActiveSliceIterator(const ActiveSlice* slc, size_t ix = 0) : elem{nullptr}, iterationIndex{ix}, container{slc->m_cont}, iterIndices{&slc->indices} {
            setPointer(ix);
        }

        reference operator*() { return *elem; }
        pointer operator->() { return elem; }

        ActiveSliceIterator& operator++() {
            setPointer(++iterationIndex);
            return *this;
        }

        ActiveSliceIterator operator++(int) {
            ActiveSliceIterator temp = *this;
            ++(*this);
            return temp;
        }

        ActiveSliceIterator& operator--() {
            setPointer(--iterationIndex);
            return *this;
        }

        ActiveSliceIterator operator--(int) {
            ActiveSliceIterator temp = *this;
            setPointer(--iterationIndex);
            return temp;
        }

        ActiveSliceIterator& operator+=(difference_type n) {
            setPointer(iterationIndex + n);
            iterationIndex += n;
            return *this;
        }

        ActiveSliceIterator& operator-=(difference_type n) {
            setPointer(iterationIndex - n);
            iterationIndex -= n;
            return *this;
        }

        ActiveSliceIterator operator+(difference_type n) const {
            ActiveSliceIterator temp = *this;
            temp.setPointer(temp.iterationIndex + n);
            return temp;
        }

        ActiveSliceIterator operator-(difference_type n) const {
            ActiveSliceIterator temp = *this;
            temp.setPointer(temp.iterationIndex - n);
            return temp;
        }

        difference_type operator-(const ActiveSliceIterator& other) const {
            return static_cast<difference_type>(iterationIndex - other.iterationIndex);
        }

        bool operator==(const ActiveSliceIterator& other) const { return iterationIndex == other.iterationIndex; }
        bool operator!=(const ActiveSliceIterator& other) const { return !(*this == other); }
        bool operator<(const ActiveSliceIterator& other) const { return iterationIndex < other.iterationIndex; }
        bool operator>(const ActiveSliceIterator& other) const { return iterationIndex > other.iterationIndex; }
        bool operator<=(const ActiveSliceIterator& other) const { return iterationIndex <= other.iterationIndex; }
        bool operator>=(const ActiveSliceIterator& other) const { return iterationIndex >= other.iterationIndex; }

    };

    using iterator = ActiveSliceIterator;

public:
    explicit ActiveSlice(ContainerType* cont) : m_cont{cont}, indices(cont->size()), activeIndex{0} {
        std::iota(indices.begin(), indices.end(), 0);
    }

    ActiveSliceIterator begin() {
        return ActiveSliceIterator{this};
    }

    ActiveSliceIterator end() {
        return ActiveSliceIterator{this, activeIndex};
    }

    ActiveSliceIterator begin() const {
        return ActiveSliceIterator{this};
    }

    ActiveSliceIterator end() const {
        return ActiveSliceIterator{this, activeIndex};
    }

    void insert(T&& elem, bool active=true) {
        size_t nextIx = m_cont->size();
        m_cont->push_back(std::move(elem));
        indices.push_back(nextIx);
        if(active) {
            std::swap(indices.back(), indices[activeIndex]);
            ++activeIndex;
        }
    }

    /**
     * Inserts an element as either active or inactive. Guarantees that the given iterator it remains valid and points
     * to the same element after. Guarantees that all elements which were active before the call and were reachable (via increment)
     * from the given iterator it, still remain so after the call. If an active element is inserted, it will not be
     * reachable (via increment) from the iterator it. Other iterators may be invalidated! Runs in (amortized) constant time.
     * @param elem The element to move-construct in the maintained list
     * @param it An iterator pointing to an active element
     * @param active whether the element is active (default: true)
     */
    void insert(T&& elem, iterator& it, bool active=true) {
        size_t currentIx = it.iterationIndex;
        assert(currentIx < activeIndex);
        size_t nextIx = m_cont->size();
        m_cont->push_back(std::move(elem));
        indices.push_back(nextIx); // may invalidate the given iterator it (and other iterators)
        if(active) {
            std::swap(indices.back(), indices[activeIndex]);
            std::swap(indices[currentIx+1], indices[activeIndex]);
            std::swap(indices[currentIx+1], indices[currentIx]);
            ++activeIndex;
            it = iterator{this, currentIx + 1};
        } else {
            it = iterator{this, currentIx};
        }
        assert(m_cont->at(indices[it.iterationIndex]) == *it);
    }

    /**
     * Deactivates the element pointed to by the given iterator. Returns an iterator itx that points to an active
     * element which was reachable (via increment) from the given iterator it before the call (or end() if no such
     * element exists). It is further guaranteed that all elements which were active before the call and were reachable
     * (via increment) from the given iterator it, remain active and reachable (via increment) from itx after the call.
     * The given iterator it points to the same element after, but note that modifying an iterator to an inactive
     * element is undefined behavior.
     *
     * Runs in constant time.
     * @param it An iterator pointing to an active element
     */
    iterator deactivate(iterator it) {
        size_t ix = it.iterationIndex;
        std::swap(indices[ix], indices[activeIndex-1]);
        --activeIndex;
        it = iterator{this, ix};
        assert((*m_cont)[indices[it.iterationIndex]] == *it);
        return it;
    }

    void activateAll() {
        activeIndex = m_cont->size();
    }

    [[nodiscard]] bool isActive() const {
        return activeIndex > 0;
    }

    [[nodiscard]] int numActive() const {
        return activeIndex;
    }
};

// Factory function to enable template argument deduction
template <typename T, template <typename...> class Container>
ActiveSlice<T, Container> buildActiveSlice(Container<T>& container) {
    return ActiveSlice<T, Container>(&container);
}

#endif //RCUT_ACTIVE_SLICE_H
