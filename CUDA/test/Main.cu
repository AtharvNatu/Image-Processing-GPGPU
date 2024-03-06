#include "../include/ChangeDetection/CudaChangeDetection.cuh"

int gpuChoice = -1;

void printDeviceProperties(void)
{
	// Code
	std::cout << std::endl << "Detected Nvidia GPU ... Using CUDA ...";
	std::cout << std::endl << "-----------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl << "CUDA INFORMATION : " << std::endl;
	std::cout << std::endl << "***********************************************************************************";
	
	cudaError_t retCudaRt;
	int devCount;

	retCudaRt = cudaGetDeviceCount(&devCount);

	if (retCudaRt != cudaSuccess)
	{
		std::cout << std::endl << "CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To " << cudaGetErrorString(retCudaRt) << std::endl;
	}
	else if (devCount == 0)
	{
		std::cout << std::endl << "No CUDA Supported Devices Found On This System ... Exiting !!!" << std::endl;
		return;
	}
	else
	{
		for (int i = 0; i < devCount; i++)
		{
			cudaDeviceProp devProp;
			int runtimeVersion = 0;

			retCudaRt = cudaGetDeviceProperties(&devProp, i);
			if (retCudaRt != cudaSuccess)
			{
				std::cout << std::endl << " " << cudaGetErrorString(retCudaRt) << "in" << __FILE__ << "at line " << __LINE__ << std::endl;
				return;
			}

			cudaRuntimeGetVersion(&runtimeVersion);

			std::cout << std::endl << "GPU Device Number			: " << i;
			std::cout << std::endl << "GPU Device Name				: " << devProp.name;
			std::cout << std::endl << "CUDA Version				: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10;
			std::cout << std::endl << "GPU Device Memory			: " << ceil(((float)devProp.totalGlobalMem / 1048576.0f) / 1024.0f) << " GB";
			std::cout << std::endl << "GPU Device Number Of SMProcessors	: " << devProp.multiProcessorCount;
		}

		// GPU Selection
		if (devCount > 1)
		{
			std::cout << std::endl << "You have more than 1 CUDA GPU Devices ... Please select 1 of them";
			std::cout << std::endl << "Enter GPU Device Number : ";
			std::cin >> gpuChoice;

			// Set CUDA GPU Device
			cudaSetDevice(gpuChoice);
		}
		else
		{
			// Set CUDA GPU Device
			cudaSetDevice(0);
		}

		std::cout << std::endl << "***********************************************************************************";
		std::cout << std::endl << "-----------------------------------------------------------------------------------" << std::endl;
	}
}



int main(int argc, char **argv)
{
    // Code
    CudaChangeDetection *cudaChangeDetector = new CudaChangeDetection();

    std::cout << std::endl << "--------------------" << std::endl << "DEBUG MODE" << std::endl << "--------------------" << std::endl;

    // printDeviceProperties();

    double gpuTime = cudaChangeDetector->detectChanges(
        std::string(argv[1]),
        std::string(argv[2]),
        std::string(argv[3]),
        true
    );
        
    std::cout << std::endl << "Time Required Using CUDA : "<< gpuTime << " seconds" << std::endl;
    
    delete cudaChangeDetector;
    cudaChangeDetector = nullptr;

    return 0;
}
