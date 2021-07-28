/* The module transforms neighbors in counterclockwise manner.
 * Author: Artem Komarichev
 * All Rights Reserved. 2018.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>

//m:number of points，n: dimension，m_q: number of points after fps,k: number of neighboring points，num_k: number of kernel points,d: dimension of kernel points
__global__ void transform_neighbors_gpu(int b, int m, int n, int m_q, int k,int num_k,int d, float sita,int interp,int fit,float radius, const float *input, const float *queries,
 const float *queries_norm, const int *idx,const float *kernel,const float * axis_x,const float * axis_y, float *proj, int *outi, float *angles,float *kernel_out,float *weight,float *kernel_fit){
  int batch_index = blockIdx.x;
  queries+=m_q*n*batch_index; //(512*3*16)
  queries_norm+=m_q*n*batch_index;
  axis_x+=m_q*n*batch_index;
  axis_y+=m_q*n*batch_index;
  idx+=m_q*k*batch_index; //(16,512,16)
  angles+=m_q*k*batch_index;
  input+=m*n*batch_index;
  proj+=m_q*k*n*batch_index;
  weight+=m_q*k*num_k*batch_index;
  kernel_fit+=m_q*num_k*4*batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;


  for(int i=index; i<m_q; i+=stride){
    float dist_max=0;
    float dist_pro=0;
    // 1.  calculte the projection by normal (n_x, n_q, n_z)
    for(int j=0; j<k; ++j){
      int pnt_idx = idx[i*k+j];
      float x_p = input[pnt_idx*n + 0];
      float y_p = input[pnt_idx*n + 1];
      float z_p = input[pnt_idx*n + 2];
      //calculate the relative coordinates
      float x_q = queries[i*n + 0];
      float y_q = queries[i*n + 1];
      float z_q = queries[i*n + 2];
      float rela_x = x_p - x_q;
      float rela_y = y_p - y_q;
      float rela_z = z_p - z_q;
        // project if fit==1, real otherwise
      if(1){//if(fit==1){
      float n_x = queries_norm[i*n + 0];
      float n_y = queries_norm[i*n + 1];
      float n_z = queries_norm[i*n + 2];
      float x_x = axis_x[i*n + 0];
      float x_y = axis_x[i*n + 1];
      float x_z = axis_x[i*n + 2];
      float y_x = axis_y[i*n + 0];
      float y_y = axis_y[i*n + 1];
      float y_z = axis_y[i*n + 2];


      // calculate the transformed coordinates of neighboring points
      proj[(i*k+j)*n + 0] = x_x* rela_x + x_y*rela_y + x_z*rela_z;
      proj[(i*k+j)*n + 1] = y_x* rela_x + y_y*rela_y + y_z*rela_z;
      proj[(i*k+j)*n + 2] = n_x* rela_x + n_y*rela_y + n_z*rela_z;
      dist_pro=sqrt(proj[(i*k+j)*n + 0]*proj[(i*k+j)*n + 0]+proj[(i*k+j)*n + 1]*proj[(i*k+j)*n + 1]);
      }
      else{
      proj[(i*k+j)*n + 0] = rela_x;
      proj[(i*k+j)*n + 1] = rela_y;
      proj[(i*k+j)*n + 2] = rela_z;
      dist_pro=sqrt(rela_x*rela_x+rela_y*rela_y+rela_z*rela_z);
      }
      if (dist_pro> dist_max) dist_max=dist_pro;
      //for test
      kernel_out[0]=proj[(i*k+j)*n + 0];
      kernel_out[1]=proj[(i*k+j)*n + 1];
      kernel_out[2]=proj[(i*k+j)*n + 2];
      kernel_out[3]=rela_x;
      kernel_out[4]=rela_y;
      kernel_out[5]=rela_z;
      //kernel_out[6]=n_x;
      //kernel_out[7]=n_y;
      //kernel_out[8]=n_z;
    }
    //utilize the maximum radius or the input radius
    if (radius) dist_max=radius;
    // calculate the distance and the coordinates of the center point
    for(int m=0; m<num_k; m++){
      //count the number of neighboring points for each kernel point
      int cnt=0;
      for(int j=0; j<k; ++j){
      float x_p = proj[(i*k+j)*n+0];
      float y_p = proj[(i*k+j)*n+1];
      float z_p = proj[(i*k+j)*n+2];
      float dist=0;
      if (d==2){
      dist=max(sqrtf((x_p-kernel[m*2]*dist_max)*(x_p-kernel[m*2]*dist_max)+(y_p-kernel[m*2+1]*dist_max)*(y_p-kernel[m*2+1]*dist_max)),1e-20f);
      kernel_out[13]=12;
      }
      else if (d==3){
      dist=max(sqrtf((x_p-kernel[m*3]*dist_max)*(x_p-kernel[m*3]*dist_max)+(y_p-kernel[m*3+1]*dist_max)*(y_p-kernel[m*3+1]*dist_max)+(z_p-kernel[m*3+2]*dist_max)*(z_p-kernel[m*3+2]*dist_max)),1e-20f);
      }
      if(dist<sita*dist_max){
      kernel_fit[(i*num_k+m)*4+0]+=x_p;
      kernel_fit[(i*num_k+m)*4+1]+=y_p;
      kernel_fit[(i*num_k+m)*4+2]+=z_p;
      cnt+=1;
      weight[(i*k+j)*num_k+m]=1;
      }
      }

      if(cnt!=0){
      float x_k=0;
      float y_k=0;
      float z_k=0;
      kernel_fit[(i*num_k+m)*4+3]=cnt;
      if(fit){
      x_k=kernel_fit[(i*num_k+m)*4+0]/cnt;
      y_k=kernel_fit[(i*num_k+m)*4+1]/cnt;
      z_k=kernel_fit[(i*num_k+m)*4+2]/cnt;
      }
      else{
      x_k=kernel[m*3]*dist_max;
      y_k=kernel[m*3+1]*dist_max;
      z_k=kernel[m*3+2]*dist_max;
      }
      //calculate the weight between neighboring point j and kernel point m
      for(int j=0; j<k; ++j){
        if(weight[(i*k+j)*num_k+m]==1){
        float x_p;
        float y_p;
        float z_p;
        x_p = proj[(i*k+j)*n+0];
        y_p = proj[(i*k+j)*n+1];
        z_p = proj[(i*k+j)*n+2];

        if (interp==1){
            float dist=max(sqrtf((x_p-x_k)*(x_p-x_k)+(x_p-x_k)*(y_p-y_k)+(y_p-y_k)*(z_p-z_k)),1e-20f);
            weight[(i*k+j)*num_k+m]=1-dist/(sita*dist_max);
        }
        else if (interp==0){
            weight[(i*k+j)*num_k+m]=max((1-abs(x_p-x_k)/(sita*dist_max))*(1-abs(y_p-y_k)/(sita*dist_max))*(1-abs(z_p-z_k)/(sita*dist_max)),1e-20f);
        }
        else if (interp==2){
            weight[(i*k+j)*num_k+m]=max((1-abs(x_p-x_k)/(sita*dist_max))*(1-abs(y_p-y_k)/(sita*dist_max))*(1-abs(z_p-z_k)/(sita*dist_max)),1e-20f);
            weight[(i*k+j)*num_k+m]/=cnt;
        }
      }

      }
      kernel_out[12]=dist_max;
      kernel_out[14]=14;

      }
    }
    kernel_out[9]=weight[1234];
    kernel_out[10]=weight[345];
    kernel_out[11]=weight[453];
  }
}


void transformNeighborsLauncher(int b, int m, int n, int m_q, int k,int num_k,int d, float sita,int interp,int fit,float radius, const float *input, const float *queries,
 const float *queries_norm, const int *idx, const float *kernel,const float * axis_x,const float * axis_y,float *proj, int *outi, float *angles,float *kernel_out,float *weight,float *kernel_fit){
  transform_neighbors_gpu<<<b,256>>>(b,m,n,m_q,k,num_k,d,sita,interp,fit,radius, input,queries,queries_norm,idx,kernel,axis_x,axis_y,proj,outi,angles,kernel_out,weight,kernel_fit);
  cudaDeviceSynchronize();
}
