#!/usr/bin/env bash
#/bin/bash
#/usr/local/cuda-8.0/bin/nvcc tf_transforming_g.cu -o tf_transforming_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
CUDA_PATH=/usr/local/cuda-10.0
$CUDA_PATH/bin/nvcc tf_transforming_g.cu -o tf_transforming_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

#g++ -std=c++11 tf_transforming.cpp tf_transforming_g.cu.o -o tf_transforming_so.so -shared -fPIC -I /../anaconda3/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_transforming.cpp tf_transforming_g.cu.o -o tf_transforming_so.so -shared -fPIC -fPIC -I $TF_INC -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/  -L$TF_LIB -I$TF_INC/external/nsync/public/ -ltensorflow_framework  -O2 -D_GLIBCXX_USE_CXX11_ABI=0
