REM Set cache dir and copy from master
set COMPILEDIR=C:\\Jenkins\\theano_cache\\pr_win
C:\Windows\System32\robocopy /E/purge C:\Jenkins\theano_cache\buildbot_windows C:\Jenkins\theano_cache\pr_win > nul

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
C:\Windows\System32\robocopy /E libgpuarray C:\Jenkins\lib\pr_win\libgpuarray > nul
set PATH=%PATH%;C:\Jenkins\lib\pr_win\libgpuarray\lib;C:\lib\cuda\bin

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


set THEANO_PARAM=theano --with-timer --timer-top-n 10 --with-xunit --xunit-file=theano_win_pr_tests.xml
set NAME=pr_win
set THEANO_FLAGS=%THEANO_FLAGS%,mode=FAST_RUN,floatX=float32,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800,compiledir=%COMPILEDIR%,dnn.include_path=C:\\lib\\cuda\\include,dnn.library_path=C:\\lib\\cuda\\lib\\x64,gcc.cxxflags='-I"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include" -I"C:\\Jenkins\\lib\\pr_win\\libgpuarray\\src" -L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v8.0\\lib\\x64" -LC:\\Jenkins\\lib\\pr_win\\libgpuarray\\lib'
python bin\theano-nose %THEANO_PARAM% --xunit-testsuite-name=%NAME%