/* The module orders neighbors in counterclockwise manner.
 * Author: Artem Komarichev
 * All Rights Reserved. 2018.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;
using namespace std;

REGISTER_OP("TransformNeighbors")
  .Attr("k: int")
  .Attr("sita: float")
  .Attr("interp: int")
  .Attr("fit: int")
  .Attr("radius: float")
  .Input("input_xyz: float32")
  .Input("query_xyz: float32")
  .Input("query_normals: float32")
  .Input("idx: int32")
  .Input("kernel: float32")
  .Input("axis_x: float32")
  .Input("axis_y: float32")
  .Output("outi: int32")
  .Output("proj: float32")
  .Output("angles: float32")
  .Output("kernel_out: float32")
  .Output("weight: float32")
  .Output("kernel_fit: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    int k;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
    float sita;
    TF_RETURN_IF_ERROR(c->GetAttr("sita", &sita));
    int interp;
    TF_RETURN_IF_ERROR(c->GetAttr("interp", &interp));
    int fit;
    TF_RETURN_IF_ERROR(c->GetAttr("fit", &fit));
    float radius;
    TF_RETURN_IF_ERROR(c->GetAttr("radius", &radius));
    ::tensorflow::shape_inference::ShapeHandle dims2;
    c->WithRank(c->input(3), 3, &dims2);
    ::tensorflow::shape_inference::ShapeHandle dims1;
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k, c->Dim(dims1,2)});
    c->set_output(0, c->input(3));
    c->set_output(1, output2);
    c->set_output(2, c->input(3));
    ::tensorflow::shape_inference::ShapeHandle dims3;
    c->WithRank(c->input(4), 2, &dims3);
    ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims3, 0), 3});
    c->set_output(3, output3);
    ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k , c->Dim(dims3, 0)});
    c->set_output(4, output4);
    ::tensorflow::shape_inference::ShapeHandle output5 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims3, 0),4});
    c->set_output(5, output5);
    return Status::OK();
  });

void transformNeighborsLauncher(int b, int m, int n, int m_q, int k,int num_k,int d,float sita,int interp,int fit,float radius,const float *input, const float *queries, const float *queries_norm,
 const int *idx, const float *kernel,const float * axis_x,const float * axis_y, float *proj, int *outi, float *angles, float *kernel_out,float *weight,float *kernel_fit);
class TransformNeighborsGpuOp : public OpKernel {
      public:
          explicit TransformNeighborsGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("TransformNeighbors expects positive k"));
            OP_REQUIRES_OK(context, context->GetAttr("sita", &sita_));
            OP_REQUIRES(context, sita_ > 0, errors::InvalidArgument("TransformNeighbors expects positive sita"));
            OP_REQUIRES_OK(context, context->GetAttr("interp", &interp_));
            OP_REQUIRES_OK(context, context->GetAttr("fit", &fit_));
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));

          }

          void Compute(OpKernelContext* context) override {
            const Tensor& input_xyz_tensor = context->input(0);
            OP_REQUIRES(context, input_xyz_tensor.dims() == 3, errors::InvalidArgument("TransformNeighbors expects (b,m,n) input_xyz shape"));

            int b = input_xyz_tensor.shape().dim_size(0);
            int m = input_xyz_tensor.shape().dim_size(1);
            int n = input_xyz_tensor.shape().dim_size(2);//n是特征维度？ //3

            const Tensor& query_xyz_tensor = context->input(1);
            OP_REQUIRES(context, query_xyz_tensor.dims() == 3, errors::InvalidArgument("TransformNeighbors expects (b,m_q,n) query_xyz shape"));

            int m_q = query_xyz_tensor.shape().dim_size(1);

            const Tensor& query_normals_tensor = context->input(2);
            OP_REQUIRES(context, query_normals_tensor.dims() == 3, errors::InvalidArgument("TransformNeighbors expects (b,m_q,n) query_normals shape"));

            const Tensor& idx_tensor = context->input(3);
            OP_REQUIRES(context, idx_tensor.dims() == 3, errors::InvalidArgument("TransformNeighbors expects (b,m_q,k) idx shape"));

            int k = idx_tensor.shape().dim_size(2);
            //我加的
            const Tensor& kernel_tensor = context->input(4);
            //OP_REQUIRES(context, kernel_tensor.dims() == 2, errors::InvalidArgument("TransformNeighbors expects (num_k,2) kernel shape"));
            int num_k=kernel_tensor.shape().dim_size(0);
            int d=kernel_tensor.shape().dim_size(1);

            const Tensor& axis_x_tensor = context->input(5);
            OP_REQUIRES(context, axis_x_tensor.dims() == 3, errors::InvalidArgument("TransformNeighbors expects (b,m_q,n) axis_x shape"));
            const Tensor& axis_y_tensor = context->input(6);
            OP_REQUIRES(context, axis_y_tensor.dims() == 3, errors::InvalidArgument("TransformNeighbors expects (b,m_q,n) axis_y shape"));

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m_q,k}, &outi_tensor));
            Tensor *proj_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m_q,k,n}, &proj_tensor));
            Tensor *angles_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,m_q,k}, &angles_tensor));
            Tensor *outkernel_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{num_k,3}, &outkernel_tensor));
            Tensor *weight_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{b,m_q,k,num_k}, &weight_tensor));
            Tensor *kernel_fit_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape{b,m_q,num_k,4}, &kernel_fit_tensor));


            auto input_flat = input_xyz_tensor.flat<float>();
            const float *input = &(input_flat(0));
            auto queries_flat = query_xyz_tensor.flat<float>();
            const float *queries = &(queries_flat(0));
            auto queries_norm_flat = query_normals_tensor.flat<float>();
            const float *queries_norm = &(queries_norm_flat(0));
            auto axis_x_flat = axis_x_tensor.flat<float>();
            const float *axis_x = &(axis_x_flat(0));
            auto axis_y_flat = axis_y_tensor.flat<float>();
            const float *axis_y = &(axis_y_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto proj_flat = proj_tensor->flat<float>();
            float *proj = &(proj_flat(0));
            cudaMemset(proj, 0.0, sizeof(float)*b*m_q*k*n);
            auto angles_flat = angles_tensor->flat<float>();
            float *angles = &(angles_flat(0));
            cudaMemset(angles, 0.0, sizeof(float)*b*m_q*k);
            //我加的
            auto kernel_flat = kernel_tensor.flat<float>();
            const float *kernel = &(kernel_flat(0));
            auto kernel_out_flat = outkernel_tensor->flat<float>();
            float *kernel_out = &(kernel_out_flat(0));
            cudaMemset(kernel_out, 0.0, sizeof(float)*num_k*3);
            auto weight_flat = weight_tensor->flat<float>();
            float *weight = &(weight_flat(0));
            cudaMemset(weight, 0.0, sizeof(float)*b*m_q*k*num_k);
            auto kernel_fit_flat = kernel_fit_tensor->flat<float>();
            float *kernel_fit = &(kernel_fit_flat(0));
            cudaMemset(kernel_fit, 0.0, sizeof(float)*b*m_q*num_k*4);

            transformNeighborsLauncher(b, m, n, m_q, k,num_k,d,sita_,interp_,fit_,radius_, input, queries, queries_norm, idx, kernel,axis_x,axis_y, proj, outi, angles, kernel_out,weight,kernel_fit);
          }

        private:
          int k_;
          float sita_;
          int interp_;
          int fit_;
          float radius_;
};
REGISTER_KERNEL_BUILDER(Name("TransformNeighbors").Device(DEVICE_GPU), TransformNeighborsGpuOp);
