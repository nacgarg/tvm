#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {
namespace transform {

// This transform pass lowers image operators to the equivalent Relay ops.
class LowerImageOpsTransform : public ExprMutator {
 public:
  explicit LowerImageOpsTransform() {}

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) final {
    // If the call is not a sobel operator, then we do not need to do anything.
    // Otherwise, create a constant weight matrix of size kernel_size for the Sobel operator
    // and use convolution ops to implement the Sobel operator.
    if (call->op.as<OpNode>()) {
      std::cout << call->op.as<OpNode>()->name << std::endl;
    }
    if (call->op.as<OpNode>() && call->op.as<OpNode>()->name == "image.sobel") {
      // Mutate the call since we're modifying the AST

      const auto* param = call->attrs.as<SobelAttrs>();
      CHECK(param != nullptr);
      const auto& kernel_size = param->kernel_size;
      const auto& kernel_height = kernel_size[0];
      const auto& kernel_width = kernel_size[1];

      const int kernel_height_int = kernel_height.as<IntImmNode>()->value;
      const int kernel_width_int = kernel_width.as<IntImmNode>()->value;

      // The kernel is a square matrix that can be 3x3, 5x5, or 7x7.

      CHECK(kernel_height_int == kernel_width_int)
          << "Sobel operator kernel must be a square matrix";
      CHECK(kernel_height_int == 3 || kernel_height_int == 5 || kernel_height_int == 7)
          << "Sobel operator kernel must be 3x3, 5x5, or 7x7";

      // Create a constant 2D tensor of size kernel_height x kernel_height for the Sobel operator.
      // For the 3x3 kernel, the weight matrix is:
      // [[-1, 0, 1],
      //  [-2, 0, 2],
      //  [-1, 0, 1]]
      // The 5x5 sobel operator is:
      // [[-5, -4, 0, 4, 5],
      //  [-8, -10, 0, 10, 8],
      //  [-10, -20, 0, 20, 10],
      //  [-8, -10, 0, 10, 8],
      //  [-5, -4, 0, 4, 5]]
      // The 7x7 sobel operator is:
      // [[-21, -18, -15, 0, 15, 18, 21],
      //  [-36, -42, -45, 0, 45, 42, 36],
      //  [-49, -60, -65, 0, 65, 60, 49],
      //  [-60, -80, -85, 0, 85, 80, 60],
      //  [-49, -60, -65, 0, 65, 60, 49],
      //  [-36, -42, -45, 0, 45, 42, 36],
      //  [-21, -18, -15, 0, 15, 18, 21]]
      // The weight matrix is a constant tensor that is used to compute the Sobel operator.

      std::vector<int64_t> weight_shape = {1, 1, kernel_height_int, kernel_height_int};
      auto weight_data = std::vector<float>(kernel_height_int * kernel_height_int, 0.0);
      if (kernel_height_int == 3) {
        weight_data = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
      } else if (kernel_height_int == 5) {
        weight_data = {-5.0f, -4.0f,  0.0f,   4.0f,  5.0f,  -8.0f, -10.0f, 0.0f,   10.0f,
                       8.0f,  -10.0f, -20.0f, 0.0f,  20.0f, 10.0f, -8.0f,  -10.0f, 0.0f,
                       10.0f, 8.0f,   -5.0f,  -4.0f, 0.0f,  4.0f,  5.0f};
      } else if (kernel_height_int == 7) {
        weight_data = {-21.0f, -18.0f, -15.0f, 0.0f,   15.0f,  18.0f,  21.0f,  -36.0f, -42.0f,
                       -45.0f, 0.0f,   45.0f,  42.0f,  36.0f,  -49.0f, -60.0f, -65.0f, 0.0f,
                       65.0f,  60.0f,  49.0f,  -60.0f, -80.0f, -85.0f, 0.0f,   85.0f,  80.0f,
                       60.0f,  -49.0f, -60.0f, -65.0f, 0.0f,   65.0f,  60.0f,  49.0f,  -36.0f,
                       -42.0f, -45.0f, 0.0f,   45.0f,  42.0f,  36.0f,  -21.0f, -18.0f, -15.0f,
                       0.0f,   15.0f,  18.0f,  21.0f};
      }
      // Create Constant tensor for the weight matrix.
      auto weight = MakeConstantTensor(DataType::Float(32), weight_shape, weight_data);
      
      auto create_conv = [&](auto input, auto weight) {
        // Replace the call node with a convolution node.
        auto conv_attrs = make_object<Conv2DAttrs>();
        conv_attrs->strides = {1, 1};
        conv_attrs->padding = {0, 0};
        conv_attrs->dilation = {1, 1};
        conv_attrs->groups = 1;
        // Set channels to 1 for the Sobel operator.
        conv_attrs->channels = 1;
        // Set kernel size in OIHW format.
        // conv_attrs->kernel_size = {1, 1, kernel_height_int, kernel_width_int};
        conv_attrs->kernel_size = {kernel_height, kernel_width};
        conv_attrs->data_layout = param->data_layout;
        conv_attrs->kernel_layout = "OIHW";
        conv_attrs->out_layout = param->data_layout;
        conv_attrs->out_dtype = param->out_dtype;
        const Op& conv2d = Op::Get("nn.conv2d");
        return Call(conv2d, {input, weight}, Attrs(conv_attrs), {});
      };

      auto x_conv = create_conv(call->args[0], weight);

      // Transpose weights using a transpose node.
      const Op& transpose = Op::Get("transpose");
      auto weight_transpose_attrs = make_object<TransposeAttrs>();
      weight_transpose_attrs->axes = {0, 1, 3, 2};
      Expr weight_transpose = Call(transpose, {weight}, Attrs(weight_transpose_attrs), {});

      auto y_conv = create_conv(call->args[0], weight_transpose);

      // gx2 = x_conv ** 2
      const Op& power = Op::Get("power");

      auto gx2 = Call(power, {x_conv, MakeConstantScalar(DataType::Float(32), 2)}, {}, {});
      auto gy2 = Call(power, {y_conv, MakeConstantScalar(DataType::Float(32), 2)}, {}, {});

      const Op& add = Op::Get("add");
      auto gxgy = Call(add, {gx2, gy2}, {}, {});

      // g = sqrt(gx2 + gy2)
      const Op& sqrt = Op::Get("sqrt");
      auto g = Call(sqrt, {gxgy}, {}, {});

      return g;
    }
    // If the call node is a image.gaussian_blur
    else if (call->op.as<OpNode>() && call->op.as<OpNode>()->name == "image.gaussian_blur") {
      // Get the kernel size from the attributes.
      const auto& param = call->attrs.as<GaussianBlurAttrs>();
      const int kernel_height = param->kernel_size[0].as<IntImmNode>()->value;
      const int kernel_width = param->kernel_size[1].as<IntImmNode>()->value;

      // Get the input shape.
      const int num_channels = param->channels;

      // Create a weight tensor with the same shape as the input tensor.
      std::vector<int64_t> weight_shape = {num_channels, num_channels, kernel_height, kernel_width};
      auto weight_data =
          std::vector<float>(num_channels * num_channels * kernel_height * kernel_width, 0.0);

      const double sigma = param->sigma;
      // Generate Gaussian blur kernel with the given kernel size and provided sigma
      // for the given number of channels

      auto gaussian_kernel = [&](int i, int j, int kernel_height, int kernel_width,
                                 double sigma) -> float {
        // If sigma is less than 1,
        return std::exp(-((i - kernel_height / 2) * (i - kernel_height / 2) +
                          (j - kernel_width / 2) * (j - kernel_width / 2)) /
                        (2 * sigma * sigma)) /
               (2 * M_PI * sigma * sigma);
      };

      for (int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
          for (int c = 0; c < num_channels; c++) {
            weight_data[c * num_channels * kernel_height * kernel_width +
                        c * kernel_height * kernel_width + i * kernel_width + j] =
                gaussian_kernel(i, j, kernel_height, kernel_width, sigma);
          }
        }
      }

      // Normalize values in kernel per channel
      for (int c = 0; c < num_channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_height; i++) {
          for (int j = 0; j < kernel_width; j++) {
            sum += weight_data[c * num_channels * kernel_height * kernel_width +
                               c * kernel_height * kernel_width + i * kernel_width + j];
          }
        }
        for (int i = 0; i < kernel_height; i++) {
          for (int j = 0; j < kernel_width; j++) {
            weight_data[c * num_channels * kernel_height * kernel_width +
                        c * kernel_height * kernel_width + i * kernel_width + j] /= sum;
          }
        }
      }

      // Create Constant tensor for the weight matrix.
      auto weight = MakeConstantTensor(DataType::Float(32), weight_shape, weight_data);

      // Create conv
      auto create_conv = [&](auto input, auto weight) {
        // Replace the call node with a convolution node.
        auto conv_attrs = make_object<Conv2DAttrs>();
        conv_attrs->strides = {1, 1};
        conv_attrs->padding = {0, 0};
        conv_attrs->dilation = {1, 1};
        conv_attrs->groups = 1;
        // Set channels to 1 for the Sobel operator.
        conv_attrs->channels = num_channels;
        // Set kernel size in OIHW format.
        conv_attrs->kernel_size = {kernel_height, kernel_width};
        conv_attrs->data_layout = param->data_layout;
        conv_attrs->kernel_layout = "OIHW";
        conv_attrs->out_layout = param->data_layout;
        conv_attrs->out_dtype = param->out_dtype;
        const Op& conv2d = Op::Get("nn.conv2d");
        return Call(conv2d, {input, weight}, Attrs(conv_attrs), {});
      };

      // Replace the call node with a convolution node.
      Expr conv = create_conv(call->args[0], weight);
      return conv;
    }
    // If the call node is a ColorSpace
    else if (call->op.as<OpNode>() && call->op.as<OpNode>()->name == "image.color_space") {
      const auto& param = call->attrs.as<ColorSpaceAttrs>();
      // Get the input shape.

      const auto& in_mode = param->in_mode;
      const auto& out_mode = param->out_mode;

      auto get_channels_from_mode = [](const std::string& mode) -> int {
        if (mode == "RGB") {
          return 3;
        } else if (mode == "BGR") {
          return 3;
        } else if (mode == "HSV") {
          return 3;
        } else if (mode == "HLS") {
          return 3;
        } else if (mode == "YUV") {
          return 3;
        } else if (mode == "GRAY") {
          return 3;
        }
        return -1;
      };

      const int num_out_channels = get_channels_from_mode(out_mode);
      const int num_in_channels = get_channels_from_mode(in_mode);

      // Handle linear color space transforms
      std::vector<std::string> linear_color_space_transforms = {"RGB", "BGR", "GRAY", "YUV"};
      if (std::find(linear_color_space_transforms.begin(), linear_color_space_transforms.end(),
                    in_mode) != linear_color_space_transforms.end()) {
        if (std::find(linear_color_space_transforms.begin(), linear_color_space_transforms.end(),
                      out_mode) != linear_color_space_transforms.end()) {
          // Create a weight matrix for the linear color space transform.

          // Create a weight tensor with the same shape as the input tensor.
          std::vector<int64_t> weight_shape = {num_in_channels, num_out_channels};
          auto weight_data = std::vector<float>(num_in_channels * num_out_channels);
          // RGB -> GRAY
          if (in_mode == "RGB" && out_mode == "GRAY") {
            weight_shape = {3, 1};
            weight_data.resize(3);
            weight_data[0] = 0.114;
            weight_data[1] = 0.587;
            weight_data[2] = 0.299;
          }
          // GRAY -> RGB
          else if (in_mode == "GRAY" && out_mode == "RGB") {
            weight_data[0] = 1.0;
            weight_data[1] = 0.0;
            weight_data[2] = 0.0;
            weight_data[3] = 0.0;
            weight_data[4] = 1.0;
            weight_data[5] = 0.0;
            weight_data[6] = 0.0;
            weight_data[7] = 0.0;
            weight_data[8] = 1.0;
          }
          // RGB -> BGR
          else if (in_mode == "RGB" && out_mode == "BGR") {
            weight_data[0] = 0.0;
            weight_data[1] = 0.0;
            weight_data[2] = 1.0;
            weight_data[3] = 0.0;
            weight_data[4] = 1.0;
            weight_data[5] = 0.0;
            weight_data[6] = 1.0;
            weight_data[7] = 0.0;
            weight_data[8] = 0.0;
          }
          // BGR -> RGB
          else if (in_mode == "BGR" && out_mode == "RGB") {
            weight_data[0] = 0.0;
            weight_data[1] = 0.0;
            weight_data[2] = 1.0;
            weight_data[3] = 0.0;
            weight_data[4] = 1.0;
            weight_data[5] = 0.0;
            weight_data[6] = 1.0;
            weight_data[7] = 0.0;
            weight_data[8] = 0.0;
          }
          // BGR -> GRAY
          else if (in_mode == "BGR" && out_mode == "GRAY") {
            weight_shape = {3, 1};
            weight_data.resize(3);
            weight_data[2] = 0.114;
            weight_data[1] = 0.587;
            weight_data[0] = 0.299;
          }
          // GRAY -> BGR
          else if (in_mode == "GRAY" && out_mode == "BGR") {
            weight_data[2] = 1.0;
            weight_data[1] = 1.0;
            weight_data[0] = 1.0;
          }
          // RGB -> YUV
          else if (in_mode == "RGB" && out_mode == "YUV") {
            weight_data[0] = 0.257;
            weight_data[1] = 0.504;
            weight_data[2] = 0.098;
            weight_data[3] = -0.148;
            weight_data[4] = -0.291;
            weight_data[5] = 0.439;
            weight_data[6] = 0.439;
            weight_data[7] = -0.368;
            weight_data[8] = -0.071;
          }
  //           Y =  0.257 * R + 0.504 * G + 0.098 * B +  16;
  // U = -0.148 * R - 0.291 * G + 0.439 * B + 128;
  // V =  0.439 * R - 0.368 * G - 0.071 * B + 128;
          // YUV -> RGB
          else if (in_mode == "YUV" && out_mode == "RGB") {
            weight_data[0] = 1.0;
            weight_data[1] = 0.0;
            weight_data[2] = 1.13983;
            weight_data[3] = 1.0;
            weight_data[4] = -0.39465;
            weight_data[5] = -0.58060;
            weight_data[6] = 1.0;
            weight_data[7] = 2.03211;
            weight_data[8] = 0.0;
          }
          // Create a weight tensor.
          auto weight = MakeConstantTensor(param->out_dtype, weight_shape, weight_data);

          // If input layout is not NHWC, then we need to transpose the input.
          Expr before_flatten = call->args[0];
          if (param->data_layout == "NCHW") {
            // Create a transpose node.
            auto transpose_attrs = make_object<TransposeAttrs>();
            transpose_attrs->axes = {0, 2, 3, 1};
            const Op& transpose = Op::Get("transpose");
            before_flatten = Call(transpose, {call->args[0]}, Attrs(transpose_attrs), {});
          }

          // Flatten all dimensions except the channel dimension.
          // Get shape tensor of the input tensor.
          const Op& reshape = Op::Get("reshape");
          auto reshape_attrs = make_object<ReshapeAttrs>();
          reshape_attrs->newshape = Array<Integer>({-1, num_in_channels});
          Expr input = Call(reshape, {before_flatten}, Attrs(reshape_attrs), {});

          // Create a matrix multiplication node.
          const Op& matmul = Op::Get("nn.matmul");
          auto matmul_attrs = make_object<MatmulAttrs>();
          matmul_attrs->transpose_a = false;
          matmul_attrs->transpose_b = false;
          Expr output = Call(matmul, {input, weight}, Attrs(matmul_attrs), {});

          if (in_mode=="RGB" && out_mode == "YUV") {
            const Op& add = Op::Get("add");
            std::vector<int64_t> add_shape = {1, 3};
            auto add_data = std::vector<float>({16, 128, 128});
            auto add_tensor = MakeConstantTensor(DataType::Float(32), add_shape, add_data);
            output = Call(add, {output, add_tensor}, Attrs(), {});
          }

          // Reshape back to the original shape.
          auto reshape_like_attrs = make_object<ReshapeLikeAttrs>();

          auto Slice2 = [](auto a, Array<Integer> begin, Array<Integer> end, int axis) {
            const Op& slice = Op::Get("strided_slice");
            auto slice_attrs = make_object<StridedSliceAttrs>();
            slice_attrs->begin = begin;
            slice_attrs->end = Array<Integer>({1});
            slice_attrs->strides = Array<Integer>({1});
            slice_attrs->axes = Array<Integer>({axis});
            slice_attrs->slice_mode = "size";
            return Call(slice, {a}, Attrs(slice_attrs), {});
          };

          const Op& reshape_like = Op::Get("reshape_like");
          if ((in_mode == "RGB" || in_mode == "BGR") && out_mode == "GRAY") {
            before_flatten = Slice2(before_flatten, {0}, {1}, 3);
          }
          output = Call(reshape_like, {output, before_flatten}, Attrs(reshape_like_attrs), {});

          if (param->data_layout == "NCHW") {
            // Create a transpose node.
            auto transpose_attrs = make_object<TransposeAttrs>();
            transpose_attrs->axes = {0, 3, 1, 2};
            const Op& transpose = Op::Get("transpose");
            output = Call(transpose, {output}, Attrs(transpose_attrs), {});
          }

          return output;
        }
      }
      if ((in_mode == "RGB" || in_mode == "BGR") && out_mode == "HSV") {
        // cmax = max(r, g, b)
        // cmin = min(r, g, b)
        // delta = cmax - cmin
        // h = 0 if delta == 0
        // h = 60 * (fmod(((G - B) / delta), 6)) if (r == cmax)
        // h = 60 * (((B - R) /  delta) + 2); if (g == cmax)
        // h = 60 * (((R - G) / delta) + 4); if (b == cmax)
        // s = 0 if cmax == 0
        // s = delta / cmax if cmax != 0
        // v = cmax
        const Op& max = Op::Get("max");
        const Op& min = Op::Get("min");
        const Op& argmax = Op::Get("argmax");
        int channel_idx = 0;
        if (param->data_layout == "NCHW") {
          channel_idx = 1;
        } else if (param->data_layout == "NHWC") {
          channel_idx = 3;
        }

        auto max_atrrs = make_object<ReduceAttrs>();
        max_atrrs->axis = {channel_idx};
        max_atrrs->keepdims = true;

        auto min_atrrs = make_object<ReduceAttrs>();
        min_atrrs->axis = {channel_idx};
        min_atrrs->keepdims = true;

        auto argmax_atrrs = make_object<ArgReduceAttrs>();
        argmax_atrrs->axis = {channel_idx};
        argmax_atrrs->keepdims = true;
        argmax_atrrs->exclude = false;
        argmax_atrrs->select_last_index = false;

        Expr cmax = Call(max, {call->args[0]}, Attrs(max_atrrs), {});
        Expr cmin = Call(min, {call->args[0]}, Attrs(min_atrrs), {});

        Expr argmaxed = Call(argmax, {call->args[0]}, Attrs(argmax_atrrs), {});

        auto delta = Subtract(cmax, cmin);
        auto h = ZerosLike(delta);
        auto s = ZerosLike(delta);
        auto v = ZerosLike(delta);

        auto Equal = [](Expr a, Expr b) {
          const Op& equal = Op::Get("equal");
          return Call(equal, {a, b}, Attrs(), {});
        };

        auto NotEqual = [](Expr a, Expr b) {
          const Op& equal = Op::Get("not_equal");
          return Call(equal, {a, b}, Attrs(), {});
        };

        auto cmaxR = Equal(argmaxed, MakeConstantScalar(DataType::Int(32), 0));
        auto cmaxG = Equal(argmaxed, MakeConstantScalar(DataType::Int(32), 1));
        auto cmaxB = Equal(argmaxed, MakeConstantScalar(DataType::Int(32), 2));

        if (in_mode == "BGR") {
          std::swap(cmaxR, cmaxB);
        }

        auto And = [](auto a, auto b) {
          const Op& and_ = Op::Get("bitwise_and");
          return Call(and_, {a, b}, Attrs(), {});
        };

        auto Slice = [](auto a, Array<Integer> begin, Array<Integer> end, int axis) {
          const Op& slice = Op::Get("strided_slice");
          auto slice_attrs = make_object<StridedSliceAttrs>();
          slice_attrs->begin = begin;
          slice_attrs->end = Array<Integer>({1});
          slice_attrs->strides = Array<Integer>({1});
          slice_attrs->axes = Array<Integer>({axis});
          slice_attrs->slice_mode = "size";
          return Call(slice, {a}, Attrs(slice_attrs), {});
        };

        auto Mod = [](auto a, auto b) {
          const Op& mod = Op::Get("floor_mod");
          return Call(mod, {a, b}, Attrs(), {});
        };

        cmaxR = And(cmaxR, NotEqual(delta, MakeConstantScalar(DataType::Float(32), 0)));
        cmaxG = And(cmaxG, NotEqual(delta, MakeConstantScalar(DataType::Float(32), 0)));
        cmaxB = And(cmaxB, NotEqual(delta, MakeConstantScalar(DataType::Float(32), 0)));

        // Slice input to get r, g, b.
        Expr R;
        Expr G;
        Expr B;

        R = Slice(call->args[0], {0}, {1}, channel_idx);
        G = Slice(call->args[0], {1}, {1}, channel_idx);
        B = Slice(call->args[0], {2}, {1}, channel_idx);

        if (in_mode == "BGR") {
          std::swap(R, B);
        }

        // h = 60 * (fmod(((G - B) / delta), 6)) if (r == cmax)
        // h = 60 * (((B - R) /  delta) + 2); if (g == cmax)
        // h = 60 * (((R - G) / delta) + 4); if (b == cmax)

        h = Where(cmaxR,
                  Multiply(MakeConstantScalar(DataType::Float(32), 60),
                           Mod(Divide(Subtract(G, B), delta),
                               MakeConstantScalar(DataType::Float(32), 6))),
                  h);
        h = Where(cmaxG,
                  Multiply(MakeConstantScalar(DataType::Float(32), 60),
                           Add(Divide(Subtract(B, R), delta),
                               MakeConstantScalar(DataType::Float(32), 2))),
                  h);
        h = Where(cmaxB,
                  Multiply(MakeConstantScalar(DataType::Float(32), 60),
                           Add(Divide(Subtract(R, G), delta),
                               MakeConstantScalar(DataType::Float(32), 4))),
                  h);

        s = Where(Equal(cmax, MakeConstantScalar(DataType::Float(32), 0)), s, Divide(delta, cmax));
        v = cmax;

        tvm::Array<Expr> args;
        args.push_back(h);
        args.push_back(s);
        args.push_back(v);

        Tuple tuple(args);

        return MakeConcatenate(tuple, channel_idx);
      }
      // Unsupported in_mode out_mode combination
      std::cout << "Unsupported in_mode : " << param->in_mode << " out_mode : " << param->out_mode
                << std::endl;
    }

    // If the call node is not an image operator, return the call node unchanged.
    return ExprMutator::VisitExpr_(call);
  };
};

Expr LowerSobelExpr(const Expr& expr) {
  VLOG_CONTEXT << "LowerSobel";
  Expr result = LowerImageOpsTransform().VisitExpr(expr);
  return result;
}
TVM_REGISTER_GLOBAL("relay._transform.LowerSobelExpr").set_body_typed(LowerSobelExpr);

Pass LowerSobel() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(LowerSobelExpr(f)); };
  return CreateFunctionPass(pass_func, 1, "LowerSobel", {});
}

TVM_REGISTER_GLOBAL("relay._transform.LowerSobel").set_body_typed(LowerSobel);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
