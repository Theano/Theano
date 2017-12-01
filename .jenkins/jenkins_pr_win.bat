REM CUDNN PATH
set CUDNNPATH=C:\lib\cuda

REM Set conda python, cudnn, cmake path
set PATH=%PATH%;C:\ProgramData\Miniconda2;C:\ProgramData\Miniconda2\Library\usr\bin;C:\ProgramData\Miniconda2\Library\bin;C:\ProgramData\Miniconda2\Scripts
set PATH=%PATH%;%CUDNNPATH%\bin;C:\Program Files\CMake\bin

REM Set cache dir and copy from master
set COMPILEDIR=%WORKSPACE%\cache
REM C:\Windows\System32\robocopy /E /purge C:\Jenkins\theano_cache\buildbot_windows %COMPILEDIR% > nul

set THEANO_FLAGS=init_gpu_device=cuda,dnn.base_path="%CUDNNPATH%"

REM Build libgpuarray
set GPUARRAY_CONFIG="Release"
set DEVICE=cuda
set LIBDIR=%WORKSPACE%\local
set PATH=%PATH%;%LIBDIR%\bin

REM Make fresh clones of libgpuarray (with no history since we dont need it)
rmdir libgpuarray /s/q
set /p GPUARRAY_BRANCH=<.jenkins/gpuarray-branch
git clone -b %GPUARRAY_BRANCH% "https://github.com/Theano/libgpuarray.git"

REM Clean up previous installs (to make sure no old files are left)
rmdir %LIBDIR% /s/q
mkdir %LIBDIR%

REM Build libgpuarray
rmdir libgpuarray\build /s/q
mkdir libgpuarray\build
cd libgpuarray\build
cmake .. -DCMAKE_BUILD_TYPE=%GPUARRAY_CONFIG% -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=%LIBDIR%
nmake
cmake --build . --target install
cd ..\..

REM Add conda gcc toolchain path
set PATH=%PATH%;C:\ProgramData\Miniconda2\Library\mingw-w64\bin

REM Build the pygpu modules
cd libgpuarray
python setup.py build_ext --inplace
mkdir %LIBDIR%\lib\python
set PYTHONPATH=%PYTHONPATH%;%LIBDIR%\lib\python
REM Then install
python setup.py install --home=%LIBDIR%
cd ..

REM Exit if theano.gpuarray import fails
python -c "import theano.gpuarray; theano.gpuarray.use('%DEVICE%')" || exit /b

set THEANO_PARAM=theano --with-timer --timer-top-n 10 --with-xunit --xunit-file=theano_win_pr_tests.xml
set NAME=pr_win
set THEANO_FLAGS=%THEANO_FLAGS%,mode=FAST_RUN,floatX=float32,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800,compiledir=%COMPILEDIR:\=\\%,gcc.cxxflags='-I%LIBDIR:\=\\%\\include -L%LIBDIR:\=\\%\\lib'
python bin\theano-nose %THEANO_PARAM% --xunit-testsuite-name=%NAME%
