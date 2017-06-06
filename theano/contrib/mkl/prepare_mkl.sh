#!/bin/sh

GetVersionName()
{
VERSION_LINE=0
if [ $1 ]; then
  VERSION_LINE=`grep __INTEL_MKL_BUILD_DATE $1/include/mkl_version.h 2>/dev/null | sed -e 's/.* //'`
fi
if [ -z $VERSION_LINE ]; then
  VERSION_LINE=0
fi
echo $VERSION_LINE  # Return Version Line
}


DST=$HOME/.local
DST_LIB=$DST/lib
DST_INC=$DST/include

if [ ! -d "$DST" ]; then
    mkdir $DST
    mkdir $DST_LIB
    mkdir $DST_INC
fi

if [ ! -d "$DST_LIB" ]; then
    mkdir $DST_LIB
fi

if [ ! -d "$DST_INC" ]; then
    mkdir $DST_INC
fi

export LD_LIBRARY_PATH=$DST_LIB:$LD_LIBRARY_PATH
export LIBRARY_PATH=$DST_LIB:$LIBRARY_PATH
export CPATH=$DST_INC:$CPATH

ICC_DIR=`which icc`
if [ ! -z $ICC_DIR ]; then
    LIB_NAME=libmklml_intel.so
else
    LIB_NAME=libmklml_gnu.so
fi

VERSION_MATCH=20170209
ARCHIVE_BASENAME=mklml_lnx_2018.0.20170425.tgz
GITHUB_RELEASE_TAG=v0.9

MKLURL="https://github.com/01org/mkl-dnn/releases/download/$GITHUB_RELEASE_TAG/$ARCHIVE_BASENAME"
MKL_CONTENT_DIR=`echo $ARCHIVE_BASENAME | rev | cut -d "." -f 2- | rev`

VERSION_LINE=`GetVersionName $MKLROOT`
if [ -z $MKLROOT ] || [ $VERSION_LINE -lt $VERSION_MATCH ]; then
    VERSION_LINE=`GetVersionName $DST`
    if [ $VERSION_LINE -lt $VERSION_MATCH ] ; then
        wget --no-check-certificate  -P $PWD $MKLURL -O $PWD/$ARCHIVE_BASENAME
        tar -xzf $PWD/$ARCHIVE_BASENAME -C $PWD

        if [ -d $PWD/$MKL_CONTENT_DIR ]; then
            cp -r $PWD/$MKL_CONTENT_DIR/lib/* $DST_LIB
            cp -r $PWD/$MKL_CONTENT_DIR/include/* $DST_INC
        else
            echo "[Error]: Failed to download or unpach mklml library"
            exit 1
        fi
    fi
    LOCALMKL=`find $DST -name $LIB_NAME 2>/dev/null`
    MKL_ROOT=`echo $LOCALMKL | sed -e 's/\/lib.*$//'`
elif [ ! -z "$MKLROOT" ]; then
    LOCALMKL=`find $MKLROOT -name $LIB_NAME 2>/dev/null`
    MKL_ROOT=`echo $LOCALMKL | sed -e 's/\/lib.*$//'`
fi

if [ -z "$LOCALMKL" ]; then
    LOCALMKL=`find $MKLROOT -name libmkl_rt.so -print -quit 2>/dev/null`
    MKL_ROOT=`echo $LOCALMKL | sed -e 's/\/lib.*$//'`
fi

if [ ! -z "$LOCALMKL" ]; then
    LIBRARIES=`basename $LOCALMKL | sed -e 's/^.*lib//' | sed -e 's/\.so.*$//'`
else
    LIBRARIES="mkl_rt"
    MKL_ROOT=$MKLROOT
fi

export MKLROOT=$MKL_ROOT
echo "Finish preparing Intel(R) mklml library for Theano..."
echo $MKL_ROOT
echo $LIBRARIES
