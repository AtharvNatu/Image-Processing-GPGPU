cls

@echo off

cd bin/

@REM For Executable
nvcc.exe --std=c++20 -c "../src/CUDA/SobelCUDA.cu" -I "C:\opencv\build\include"
nvcc.exe --std=c++20 -c "../src/CUDA/CannyCUDA.cu"  -I "C:\opencv\build\include"

cl.exe /std:c++20 /c /EHsc ^
    -I "C:\opencv\build\include" ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    "../src/OpenCV/SobelCV.cpp" ^
    "../src/OpenCV/CannyCV.cpp" ^
    "../test/Main.cpp"

link.exe Main.obj SobelCUDA.obj SobelCV.obj CannyCUDA.obj CannyCV.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" cudart.lib opencv_world480.lib opencv_world480d.lib

@move Main.exe "../" > nul
cd ../


@REM @REM For DLL
@REM nvcc.exe --std=c++20 -c "../src/CUDA/SobelCUDA.cu" -I "C:\opencv\build\include"
@REM nvcc.exe --std=c++20 -c "../src/CUDA/CannyCUDA.cu"  -I "C:\opencv\build\include"

@REM cl.exe /std:c++20 /c /EHsc ^
@REM     -I "C:\opencv\build\include" ^
@REM     -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
@REM     "../src/OpenCV/SobelCV.cpp" ^
@REM     "../src/OpenCV/CannyCV.cpp" ^
@REM     "../export/Lib.cpp"

@REM link.exe /DLL /OUT:EdgeDetection.dll Lib.obj SobelCUDA.obj SobelCV.obj CannyCUDA.obj CannyCV.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" cudart.lib opencv_world480.lib opencv_world480d.lib

@REM @move EdgeDetection.dll "../" > nul

@REM cd ../
