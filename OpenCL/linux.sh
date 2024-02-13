OPENCV_INCLUDE_PATH=/usr/include/opencv4

#? USING AMD ROCm OpenCL SDK
OPENCL_INCLUDE_PATH=/opt/rocm/include/

SHARED_LIB=false

#! g++ -Wall -c ../src/CLFW/*.cpp ../test/*.cpp -I "/opt/rocm/include/"
#! g++ -o VecAdd *.o -L "/opt/rocm/lib/" -lOpenCL -lm

if [ $SHARED_LIB == false ]
then
    clear

    cd ./bin

    # For Executable
    echo "Compiling Source Files and Linking Libraries ... "
   
    g++ -Wall -Wno-deprecated -Wno-deprecated-declarations -std=c++20 -o App \
    ../test/Main.cpp ../src/ChangeDetection/*.cpp ../src/CLFW/CLFW.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} \
    -I ${OPENCL_INCLUDE_PATH} \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lm -lOpenCL

    cp App ../
    echo "Done ... "

    cd ..

    echo "Running Executable ... "
    ./App \
    /home/atharv/Desktop/Internship/Images/Dubai_1.jpg \
    /home/atharv/Desktop/Internship/Images/Dubai_2.jpg \
    /home/atharv/Desktop/Internship/Images/ \

else
    clear

    cd ./bin

    # For Shared Object
    echo "Compiling Source Files ... "
    g++ -fPIC -Wall -Wno-deprecated -std=c++20 -c \
    ../export/Lib.cpp ../src/ChangeDetection/*.cpp ../src/CLFW/CLFW.cpp ../src/Common/*.cpp \
    -I ${OPENCV_INCLUDE_PATH} \
    -I ${OPENCL_INCLUDE_PATH} \
    -DRELEASE

    echo "Creating Shared Object ..."
    g++ -shared -o libIPUG-OpenCL.so *.o \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lOpenCL -lm

    cp libIPUG-OpenCL.so ../

    echo "Generated Shared Object : libIPUG-OpenCL.so ..." 

    cd ..
fi
