cls

@echo off

set dll=false


if %dll% == false (

    @REM For Executable
    cd bin/
    
    cl.exe /std:c++20 /c /EHsc ^
        -I "C:\opencv\build\include" ^
        "../test/Main.cpp" ^
        "../src/Common/*.cpp" ^
        "../src/ChangeDetection/*.cpp" ^
    
    link.exe /OUT:App.exe *.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world480.lib opencv_world480d.lib

    @move App.exe "../" > nul

    cd ../

    App.exe
  
) else (
   
    @REM For DLL
    cd bin/
    
    cl.exe /std:c++20 /c /EHsc ^
        -I "C:\opencv\build\include" ^
        "../export/Lib.cpp" ^
        "../src/Common/*.cpp" ^
        "../src/ChangeDetection/*.cpp" ^
    
    link.exe /DLL /OUT:IPG-CPU.dll *.obj /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world480.lib opencv_world480d.lib

    @move IPG-CPU.dll "../" > nul

    cd ../
)
