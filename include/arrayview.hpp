/** \file
 * \author Aditya Kashi
 */

#ifndef BLASTED_ARRAY_VIEW_H
#define BLASTED_ARRAY_VIEW_H

#include <type_traits>
#include <boost/align/aligned_alloc.hpp>

#ifndef CACHE_LINE_LEN
#define CACHE_LINE_LEN 64
#endif

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

/// An array type that can either wrap a memory block allocated externally or manage its own
template <typename T>
class ArrayView
{
public:
	/// Null constructor
	ArrayView() : data{nullptr}, len{0}, owner{false}
	{ }

	/// Allocation constructor
	ArrayView(const int size)
		: data{(T*)aligned_alloc(CACHE_LINE_LEN, size*sizeof(T))}, len{size}, owner{true}
	{ }

	/// Wrap constructor
	ArrayView(T *const arr, const int length) : data{arr}, len{length}, owner{false}
	{ }

	/// Move
	ArrayView(ArrayView<T>&& other) : data{other.data}, len{other.len}, owner{other.owner}
	{
		other.data = nullptr;
		other.len = 0;
		other.owner = false;
	}

	/// Copy
	ArrayView(const ArrayView<T>& other) 
		: data{(T*)aligned_alloc(CACHE_LINE_LEN, other.len*sizeof(T))}, len{other.len}, owner{true}
	{
		for(int i = 0; i < len; i++)
			data[i] = other.data[i];
	}

	/// Free if owner, otherwise null
	~ArrayView()
	{
		if(owner) {
			// this should be okay because if this is called, this object is the owner
			aligned_free(const_cast<typename std::remove_const<T>::type*>(data));
		}
		else {
			data = nullptr;
		}
		len = 0;
		owner = false;
	}

	/// Get the length of the array
	int size() const { return len; }

	/// Delete existing contents and re-allocate requested storage size
	void resize(const int size)
	{
		if(owner)
			aligned_free(const_cast<typename std::remove_const<T>::type*>(data));

		data = (T*)aligned_alloc(CACHE_LINE_LEN, size*sizeof(T));
		len = size;
		owner = true;
	}

	/// Wrap an existing block of storage, similar to the wrap constructor
	void wrap(T *const arr, const int length)
	{
		if(owner)
			aligned_free(data);
		data = arr;
		len = length;
		owner = false;
	}

	/// Wrap an existing block of storage but set this object as owner
	/** \warning Make sure to give pass a pointer allocated to the same alignment CACHE_LINE_LEN!
	 */
	void take_control(T *const arr, const int length)
	{
		if(owner)
			aligned_free(data);
		data = arr;
		len = length;
		owner = true;
	}

	/// Const accessor
	const T& operator[](const int i) const {
		return data[i];
	}

	/// Modifiable accessor
	T& operator[](const int i) {
		return data[i];
	}

private:
	T *data;
	int len;
	bool owner;
};

}
#endif
