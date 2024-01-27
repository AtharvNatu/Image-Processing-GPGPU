INCLUDE_PATH=/usr/local/Cellar/opencv/4.9.0_1/include/opencv4
LIB_PATH=/usr/local/Cellar/opencv/4.9.0_1/lib

DYLIB=false

if [ $DYLIB == false ]
then
    clear

    cd ./bin

    # For Executable
    echo "Compiling Source Files and Linking Libraries ... "
    clang++ -Wall -Wno-deprecated -std=c++20 -o App \
    ../test/Main.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${INCLUDE_PATH} -L ${LIB_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \

    cp App ../
    echo "Done ... "

    cd ..

    echo "Running Executable ... "
    ./App

else
    # For Dylib
    echo "Compiling Source Files ... "
    # clang -shared -fpic a.c b.c -o libTest.dylib
    clang++ -Wall -Wno-deprecated -std=c++20 -I ${INCLUDE_PATH} -c .../export/Lib.cpp ../src/*.cpp

    echo "Creating Dynamic Library ..."
    clang++ -shared -o libCPU.dylib *.o \
    -L ${LIB_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \

    cp libCPU.dylib ../

    echo "Generated Library : libCPU.dylib ..." 

    cd ..
fi

