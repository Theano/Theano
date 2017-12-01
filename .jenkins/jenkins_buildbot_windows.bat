REM CUDNN PATH
set CUDNNPATH=C:\lib\cuda

REM Set conda python, cudnn, cmake path
set PATH=%PATH%;C:\ProgramData\Miniconda2;C:\ProgramData\Miniconda2\Library\usr\bin;C:\ProgramData\Miniconda2\Library\bin;C:\ProgramData\Miniconda2\Scripts
set PATH=%PATH%;%CUDNNPATH%\bin;C:\Program Files\CMake\bin

set BUILDBOT_DIR=%WORKSPACE%\nightly_build
set COMPILEDIR=C:\Jenkins\theano_cache\buildbot_windows

REM Set test reports using nosetests xunit
set XUNIT=--with-xunit --xunit-file=
set SUITE=--xunit-testsuite-name=

set THEANO_PARAM=theano --with-timer --timer-top-n 10

set THEANO_FLAGS=init_gpu_device=cuda,dnn.base_path="%CUDNNPATH%"

REM Build libgpuarray
set GPUARRAY_CONFIG="Release"
set DEVICE=cuda
set LIBDIR=%WORKSPACE%\local
set PATH=%PATH%;%LIBDIR%\bin

REM Make fresh clones of libgpuarray (with no history since we dont need it)
rmdir libgpuarray /s/q
git clone "https://github.com/Theano/libgpuarray.git"

REM Clean up previous installs (to make sure no old files are left)
rmdir %LIBDIR% /s/q
mkdir %LIBDIR%

REM Build libgpuarray
mkdir libgpuarray\build
cd libgpuarray\build
cmake .. -DCMAKE_BUILD_TYPE=%GPUARRAY_CONFIG% -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=%LIBDIR%
nmake
cmake --build . --target install
cd ..\..

REM Set conda gcc path
set PATH=%PATH%;C:\ProgramData\Miniconda2\Library\mingw-w64\bin

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

REM Exit if theano.gpuarray import fails
python -c "import theano.gpuarray; theano.gpuarray.use('%DEVICE%')" || exit /b

REM Fast run and float32
set FILE=%BUILDBOT_DIR%\theano_python2_fastrun_f32_tests.xml
set NAME=win_fastrun_f32
set THEANO_FLAGS=%THEANO_FLAGS%,compiledir=%COMPILEDIR:\=\\%,mode=FAST_RUN,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,floatX=float32,gcc.cxxflags='-I%LIBDIR:\=\\%\\include -L%LIBDIR:\=\\%\\lib'
python bin\theano-nose %THEANO_PARAM% %XUNIT%%FILE% %SUITE%%NAME%
