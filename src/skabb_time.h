#ifndef H_SKABB_TIME
#define H_SKABB_TIME

#include <stddef.h> 
#include <sys/time.h>
#include <assert.h>

inline double mysecond() 
{
	struct timeval  tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp, &tzp);
    assert(i == 0);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#endif
