cls

@echo off

set dll=true

if %dll% == false (

    @REM For Executable
    cd bin/
    
    cl.exe /openmp /std:c++20 /c /EHsc ^
        -I "C:\opencv\build\include" ^
        "../test/Main.cpp" ^
        "../src/Common/*.cpp" ^
        "../src/ChangeDetection/*.cpp" ^
    
    link.exe /OUT:App.exe *.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world480.lib vcompd.lib

    @move App.exe "../" > nul

    cd ../

    App.exe ^
    ./images/input/1024_old.png ^
    ./images/input/1024_new.png ^
    ./images/output
  
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
