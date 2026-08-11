#pragma once

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
    explicit StridedIterator(Ptr<T, MemorySpace> data, SI stride = 1)
        : _data(data), _stride(stride) {}

    // dereferencing
    T& operator*() const { return *_data; }
    T* operator->() const { return _data.raw; }
    T& operator[](difference_type n) const { return *(_data + n * _stride); }

    // increment/decrement
    StridedIterator& operator++() {
        _data = _data + _stride;
        return *this;
    }
    StridedIterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
    StridedIterator& operator--() {
        _data = _data - _stride;
        return *this;
    }
    StridedIterator operator--(int) {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    // arithmetic
    StridedIterator& operator+=(difference_type n) {
        _data = _data + n * _stride;
        return *this;
    }
    StridedIterator& operator-=(difference_type n) {
        _data = _data - n * _stride;
        return *this;
    }
    StridedIterator operator+(difference_type n) const {
        auto tmp = *this;
        return tmp += n;
    }
    StridedIterator operator-(difference_type n) const {
        auto tmp = *this;
        return tmp -= n;
    }
    friend StridedIterator operator+(difference_type n, const StridedIterator& it) {
        return it + n;
    }

    // distance: number of steps from `other` to `*this` ( *this - other ). `_data.raw` is a `T*`,
    // so the subtraction is already in elements; dividing by the element stride gives the count.
    difference_type operator-(const StridedIterator& other) const {
        return ( _data.raw - other._data.raw ) / _stride;
    }

    // comparison
    bool operator==(const StridedIterator& other) const {
        return _data == other._data;
    }
    bool operator!=(const StridedIterator& other) const {
        return !(*this == other);
    }
    bool operator<(const StridedIterator& other) const {
        return _data.raw < other._data.raw;
    }
    bool operator<=(const StridedIterator& other) const {
        return !(*this > other);
    }
    bool operator>(const StridedIterator& other) const {
        return other < *this;
    }
    bool operator>=(const StridedIterator& other) const {
        return !(*this < other);
    }

private:
    Ptr<T, MemorySpace> _data;
    SI _stride = 0;
};

} // namespace sdot
