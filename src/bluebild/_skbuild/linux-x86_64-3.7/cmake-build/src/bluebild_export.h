
#ifndef BLUEBILD_EXPORT_H
#define BLUEBILD_EXPORT_H

#ifdef BLUEBILD_STATIC_DEFINE
#  define BLUEBILD_EXPORT
#  define BLUEBILD_NO_EXPORT
#else
#  ifndef BLUEBILD_EXPORT
#    ifdef bluebild_EXPORTS
        /* We are building this library */
#      define BLUEBILD_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define BLUEBILD_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef BLUEBILD_NO_EXPORT
#    define BLUEBILD_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef BLUEBILD_DEPRECATED
#  define BLUEBILD_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef BLUEBILD_DEPRECATED_EXPORT
#  define BLUEBILD_DEPRECATED_EXPORT BLUEBILD_EXPORT BLUEBILD_DEPRECATED
#endif

#ifndef BLUEBILD_DEPRECATED_NO_EXPORT
#  define BLUEBILD_DEPRECATED_NO_EXPORT BLUEBILD_NO_EXPORT BLUEBILD_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef BLUEBILD_NO_DEPRECATED
#    define BLUEBILD_NO_DEPRECATED
#  endif
#endif

#endif /* BLUEBILD_EXPORT_H */
