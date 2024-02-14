cls

@echo off

set dll=false

if %dll% == false (

    @REM For Executable
    cd bin/

    nvcc.exe -c -w --std=c++20 ^
        -I "C:\opencv\build\include" ^
        -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
        "../test/Main.cu" ^
        "../src/ChangeDetection/CudaChangeDetection.cu" ^
        "../src/ChangeDetection/OtsuBinarizerCuda.cu" ^
        "../src/ChangeDetection/ImageUtils.cpp" ^
        "../src/Common/CudaUtils.cu" ^
        "../src/Common/Logger.cpp" ^
    
    link.exe /DEBUG /OUT:App.exe *.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" opencv_world480.lib cudart.lib 

    @move App.exe "../" > nul

    cd ../

    App.exe ^
    "F:\Internship\Images\10000_old.png" ^
    "F:\Internship\Images\10000_new.png" ^
    "F:\Internship\Images"
  
) else (
   
    @REM For DLL
    cd bin/
    
    cl.exe /openmp /std:c++20 /c /EHsc ^
        -I "C:\opencv\build\include" ^
        "../export/Lib.cpp" ^
        "../src/Common/*.cpp" ^
        "../src/ChangeDetection/*.cpp" ^
        /DRELEASE
    
    link.exe /DLL /OUT:IPUG-CPU.dll *.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world480.lib vcomp.lib

    @move IPUG-CPU.dll "../" > nul

    cd ../
)
