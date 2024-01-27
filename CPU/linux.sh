INCLUDE_PATH=/usr/include/opencv4

SHARED_LIB=false

if [ $SHARED_LIB == false ]
then
    clear

    cd ./bin

    # For Executable
    echo "Compiling Source Files and Linking Libraries ... "
    g++ -Wall -Wno-deprecated -std=c++20 -o App \
    ../test/Main.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${INCLUDE_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \

    cp App ../
    echo "Done ... "

    cd ..

    echo "Running Executable ... "
    ./App /home/atharv/Pictures/Images/island.jpg

else
    # For Dylib
    echo "Compiling Source Files ... "
    clang++ -Wall -Wno-deprecated -std=c++20 -I ${INCLUDE_PATH} -c .../export/Lib.cpp ../src/*.cpp

    echo "Creating Shared Object ..."
    clang++ -shared -o libCPU.so *.o \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \

    cp libCPU.so ../

    echo "Generated Library : libCPU.so ..." 

    cd ..
fi

