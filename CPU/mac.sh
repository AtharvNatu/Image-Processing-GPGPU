OPENCV_INCLUDE_PATH=/usr/local/Cellar/opencv/4.9.0_1/include/opencv4
OPENMP_INCLUDE_PATH=/usr/local/opt/libomp/include

OPENCV_LIB_PATH=/usr/local/Cellar/opencv/4.9.0_1/lib
OPENMP_LIB_PATH=/usr/local/opt/libomp/lib

DYLIB=false

if [ $DYLIB == false ]
then
    clear

    cd ./bin

    # For Executable
    echo "Compiling Source Files and Linking Libraries ... "
    clang++ -Wall -Wno-deprecated -std=c++20 -Xclang -fopenmp -o App \
    ../test/Main.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} -I ${OPENMP_INCLUDE_PATH} \
    -L ${OPENCV_LIB_PATH} -L ${OPENMP_LIB_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lm -lomp \
    
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
    clang++ -Wall -Wno-deprecated -std=c++20 -Xclang -fopenmp -c \
    ../export/Lib.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} -I ${OPENMP_INCLUDE_PATH} \
    -DRELEASE

    echo "Creating Dynamic Library ..."
    clang++ -shared -o libIUG-CPU.dylib *.o \
    -L ${OPENCV_LIB_PATH} \
    -L ${OPENMP_LIB_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lomp

    cp libIUG-CPU.dylib ../

    echo "Generated Library : libIUG-CPU.dylib ..." 

    cd ..
fi

