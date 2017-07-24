set BUILDBOT_DIR=%WORKSPACE%\nightly_build
set COMPILEDIR=C:\\Jenkins\\theano_cache\\buildbot_windows

REM Set test reports using nosetests xunit
set XUNIT=--with-xunit --xunit-file=
set SUITE=--xunit-testsuite-name=

set THEANO_PARAM=theano --with-timer --timer-top-n 10

set THEANO_FLAGS=init_gpu_device=cuda

REM Build libgpuarray
set GPUARRAY_CONFIG="Release"
set DEVICE=cuda
set LIBDIR=%WORKSPACE%\local

REM Make fresh clones of libgpuarray (with no history since we dont need it)
rmdir libgpuarray /s/q
git clone --depth 1 "https://github.com/Theano/libgpuarray.git"

REM Clean up previous installs (to make sure no old files are left)
rmdir %LIBDIR% /s/q
mkdir %LIBDIR%

REM Build libgpuarray
set PATH=%PATH%;C:\Program Files\CMake\bin
mkdir libgpuarray\build
cd libgpuarray\build
cmake .. -DCMAKE_BUILD_TYPE=%GPUARRAY_CONFIG% -G "NMake Makefiles"
nmake
cd ..\..

REM Copy lib and export paths
C:\Windows\System32\robocopy /E libgpuarray C:\Jenkins\lib\buildbot_win\libgpuarray > nul
set PATH=%PATH%;C:\Jenkins\lib\buildbot_win\libgpuarray\lib;C:\lib\cuda\bin

REM Set conda python path
set PATH=%PATH%;C:\ProgramData\Miniconda2;C:\ProgramData\Miniconda2\Library\mingw-w64\bin;C:\ProgramData\Miniconda2\Library\usr\bin;C:\ProgramData\Miniconda2\Library\bin;C:\ProgramData\Miniconda2\Scripts

REM Build the pygpu modules
cd libgpuarray
python setup.py build_ext --inplace -I%LIBDIR%\include -L%LIBDIR%\lib
mkdir %LIBDIR%\lib\python
set PYTHONPATH=%PYTHONPATH%;%LIBDIR%\lib\python
REM Then install
python setup.py install --home=%LIBDIR%
cd ..

mkdir %BUILDBOT_DIR%
echo "Directory of stdout/stderr %BUILDBOT_DIR%"

REM Fast run and float32
set FILE=%BUILDBOT_DIR%\theano_python2_fastrun_f32_tests.xml
set NAME=fastrun_f32
set THEANO_FLAGS=%THEANO_FLAGS%,compiledir=%COMPILEDIR%,mode=FAST_RUN,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,floatX=float32,dnn.include_path=C:\\lib\\cuda\\include,dnn.library_path=C:\\lib\\cuda\\lib\\x64,gcc.cxxflags='-I"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include" -I"C:\\Jenkins\\lib\\buildbot_win\\libgpuarray\\src" -L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v8.0\\lib\\x64" -LC:\\Jenkins\\lib\\buildbot_win\\libgpuarray\\lib'
python bin\theano-nose %THEANO_PARAM% %XUNIT%%FILE% %SUITE%%NAME%
