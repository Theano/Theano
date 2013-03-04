@echo off

rem if ANACONDA_DIR is not defined
if [%ANACONDA_DIR%] == [^%ANACONDA_DIR^%] (
    if exist "c:\Anaconda" set ANACONDA_DIR=C:\Anaconda
    )

if [%ANACONDA_DIR%] == [^%ANACONDA_DIR^%] (
    echo "Anaconda not found. Please install AnacondaCE or set the ANACONDA_DIR environment variable to the location of your Anaconda installation."
    goto end
    )

if not exist %ANACONDA_DIR% (
    echo Anaconda install directory %ANACONDA_DIR% does not exist
    goto end)

echo Anaconda found in %ANACONDA_DIR%
echo copying dlls from %ANACONDA_DIR%\MinGW\x86_64-w64-mingw32\lib to %ANACONDA_DIR%\
copy %ANACONDA_DIR%\MinGW\x86_64-w64-mingw32\lib\*.dll %ANACONDA_DIR%
echo done

echo Trying to install theano
pip install Theano
echo installed

rem Put a default .theanorc.txt
set THEANORC=%USERPROFILE%\.theanorc.txt
set THEANORC_=%USERPROFILE%\.theanorc_install.txt
echo [global]> %THEANORC_%
echo openmp=False>> %THEANORC_%
echo.>> %THEANORC_%
echo [blas]>> %THEANORC_%
echo ldflags=>> %THEANORC_%

if exist %THEANORC% (
    echo A .theanorc.txt config file already exists, so we will not change it.
    echo The default version is in %THEANORC_%, we suggest you check it out.
) else (
    rename %THEANORC_% .theanorc.txt
)

:end
echo end
