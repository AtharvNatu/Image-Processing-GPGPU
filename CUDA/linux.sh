OPENCV_INCLUDE_PATH=/usr/include/opencv4
CUDA_INCLUDE_PATH=/opt/cuda/include/

SHARED_LIB=false

if [ $SHARED_LIB == false ]
then
    clear

    cd ./bin

    # For Executable
    #! -shared -Xcompiler -fPIC For Shared Object
    echo "Compiling Source Files and Linking Libraries ... "
    nvcc -g --std=c++20 -w -o App \
    ../test/Main.cu ../src/ChangeDetection/*.cu ../src/ChangeDetection/*.cpp ../src/Common/*.cpp ../src/Common/*.cu \
    -I ${OPENCV_INCLUDE_PATH} \
    -I ${CUDA_INCLUDE_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lcudart

    cp App ../
    echo "Done ... "

    cd ..

    echo "Running Executable ... "
    ./App \
    /home/atharv/Desktop/Internship/Images/40000_old.png \
    /home/atharv/Desktop/Internship/Images/40000_new.png \
    /home/atharv/Desktop/Internship/Images/ \

else
    clear

    cd ./bin

    # For Dylib
    echo "Compiling Source Files ... "
    g++ -fPIC -Wall -Wno-deprecated -std=c++20 -fopenmp -c \
    ../export/Lib.cpp ../src/ChangeDetection/*.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} \
    -DRELEASE

    echo "Creating Shared Object ..."
    g++ -shared -o libIPUG-CPU.so *.o \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lomp -lm

    cp libIPUG-CPU.dylib ../

    echo "Generated Shared Object : libIPUG-CPU.so ..." 

    cd ..
fi

