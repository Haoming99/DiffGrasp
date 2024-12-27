nvcc = /usr/local/cuda/bin/nvcc
cudalib =  /usr/local/cuda/lib64
tensorflow = /scratch/wen/miniconda3/envs/sm86/lib/python3.8/site-packages/tensorflow_core/include
TF_INC = /scratch/wen/miniconda3/envs/sm86/lib/python3.8/site-packages/tensorflow_core/include
TF_LIB = /scratch/wen/miniconda3/envs/sm86/lib/python3.8/site-packages/tensorflow_core
# gen_code_flags = -arch=sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75-gencode=arch=compute_80,code=sm_80-gencode=arch=compute_86,code=sm_86-gencode=arch=compute_87,code=sm_87 -gencode=arch=compute_86,code=compute_86
# -arch=sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75-gencode=arch=compute_80,code=sm_80-gencode=arch=compute_86,code=sm_86-gencode=arch=compute_87,code=sm_87 -gencode=arch=compute_86,code=compute_86

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -I $(TF_INC) -L $(TF_LIB) -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -arch=sm_61

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -I $(TF_INC) -L $(TF_LIB) -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2  -arch=sm_61


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
