from __future__ import absolute_import, print_function, division


def header_text():
    """
        C header for mkl dnn primitive interface,
        Build date: 20170209
    """
    header = """
    extern "C"
    {
        /* MKL Version type */
        typedef
        struct {
            int    MajorVersion;
            int    MinorVersion;
            int    UpdateVersion;
            char * ProductStatus;
            char * Build;
            char * Processor;
            char * Platform;
        } MKLVersion;

        void    MKL_Get_Version(MKLVersion *ver); /* Returns information about the version of the Intel MKL software */
        #define mkl_get_version             MKL_Get_Version
    }
    """
    return header
