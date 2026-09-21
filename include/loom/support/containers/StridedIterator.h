#pragma once

#include <loom/support/common_macros.h> // HD

#include "../kernels/Ptr.h"
#include <iterator>

namespace sdot {

/// Random-access iterator for strided data. `stride` is in ELEMENTS (of `T`), matching
/// `Ptr<T>::operator+`, which advances by elements -- NOT in bytes. `TensorView` divides its
/// byte strides by `sizeof(T)` before building the iterator.
/// Works with std algorithms; a TensorView iterates over its (innermost) axis.
template<class T, class MemorySpace>
class StridedIterator {
public:
    using difference_type = SI;
    using value_type = T;
    using pointer = Ptr<T, MemorySpace>;
    using reference = T&;
    using iterator_category = std::random_access_iterator_tag;

       StridedIterator() = default;
    HD explicit StridedIterator(Ptr<T, MemorySpace> data, SI stride = 1)
        : _data(data), _stride(stride) {}

    // dereferencing
    HD T& operator*() const { return *_data; }
    HD T* operator->() const { return _data.raw; }
    HD T& operator[](difference_type n) const { return *(_data + n * _stride); }

    // increment/decrement
    HD StridedIterator& operator++() {
        _data = _data + _stride;
        return *this;
    }
    HD StridedIterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
    HD StridedIterator& operator--() {
        _data = _data - _stride;
        return *this;
    }
    HD StridedIterator operator--(int) {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    // arithmetic
    HD StridedIterator& operator+=(difference_type n) {
        _data = _data + n * _stride;
        return *this;
    }
    HD StridedIterator& operator-=(difference_type n) {
        _data = _data - n * _stride;
        return *this;
    }
    HD StridedIterator operator+(difference_type n) const {
        auto tmp = *this;
        return tmp += n;
    }
    HD StridedIterator operator-(difference_type n) const {
        auto tmp = *this;
        return tmp -= n;
    }
    friend HD StridedIterator operator+(difference_type n, const StridedIterator& it) {
        return it + n;
    }

    // distance: number of steps from `other` to `*this` ( *this - other ). `_data.raw` is a `T*`,
    // so the subtraction is already in elements; dividing by the element stride gives the count.
    HD difference_type operator-(const StridedIterator& other) const {
        return ( _data.raw - other._data.raw ) / _stride;
    }

    // comparison
    HD bool operator==(const StridedIterator& other) const {
        return _data == other._data;
    }
    HD bool operator!=(const StridedIterator& other) const {
        return !(*this == other);
    }
    HD bool operator<(const StridedIterator& other) const {
        return _data.raw < other._data.raw;
    }
    HD bool operator<=(const StridedIterator& other) const {
        return !(*this > other);
    }
    HD bool operator>(const StridedIterator& other) const {
        return other < *this;
    }
    HD bool operator>=(const StridedIterator& other) const {
        return !(*this < other);
    }

private:
    Ptr<T, MemorySpace> _data;
    SI _stride = 0;
};

} // namespace sdot
