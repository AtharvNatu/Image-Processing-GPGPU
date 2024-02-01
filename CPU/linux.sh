OPENCV_INCLUDE_PATH=/usr/include/opencv4

SHARED_LIB=false

if [ $SHARED_LIB == false ]
then
    clear

    cd ./bin

    # For Executable
    echo "Compiling Source Files and Linking Libraries ... "
    g++ -Wall -Wno-deprecated -std=c++20 -o App \
    ../test/Main.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lm -lomp \
    -fopenmp

    cp App ../
    echo "Done ... "

    cd ..

    echo "Running Executable ... "
    ./App

else
    clear

    cd ./bin

    # For Dylib
    echo "Compiling Source Files ... "
    g++ -fPIC -Wall -Wno-deprecated -std=c++20 -fopenmp -c \
    ../export/Lib.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} \
    -DRELEASE

    echo "Creating Dynamic Library ..."
    g++ -shared -o libIUG-CPU.dylib *.o \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lomp -lm

    cp libIUG-CPU.dylib ../

    echo "Generated Library : libIUG-CPU.dylib ..." 

    cd ..
fi

