// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_NUMPY_INTERNAL_MAIN
#include <boost/numpy/internal.hpp>
#include <boost/numpy/dtype.hpp>

namespace boost 
{
namespace numpy 
{
#if PY_MAJOR_VERSION >= 3
int init_numpy() { import_array(); }
#else
void init_numpy() { import_array(); }
#endif
static void* initialize2(bool register_scalar_converters) {
	init_numpy();
	import_ufunc();
	if (register_scalar_converters)
		dtype::register_scalar_converters();
}

void initialize(bool register_scalar_converters) 
{
	initialize2( register_scalar_converters );
}

}
}
