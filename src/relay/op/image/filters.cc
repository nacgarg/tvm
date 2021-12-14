/*!
 * \file edge.cc
 * \brief Edge detection operators
 */
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.image.sobel
TVM_REGISTER_NODE_TYPE(SobelAttrs);

template <typename T>
InferCorrectLayoutOutput SobelInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Type>& new_in_types,
                                                 const Array<Type>& old_in_types,
                                                 const Array<Type>& old_out_types) {
  const auto* params = attrs.as<T>();
  return InferCorrectLayoutOutput({params->data_layout, params->kernel_layout},
                                  {params->data_layout}, attrs);
}

// Positional relay function to create sobel operator
// used by frontend FFI.
Expr MakeSobel(Expr data, Array<IndexExpr> kernel_size, String data_layout, String padding,
               DataType out_dtype) {
  auto attrs = make_object<SobelAttrs>();
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->padding = std::move(padding);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("image.sobel");
  return Call(op, {data}, Attrs(attrs), {});
}

template <typename AttrType>
bool SobelRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto dshape = data->shape;
  CHECK_EQ(dshape.size(), 4) << "Input data should be 4D: " << dshape;
  const auto* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  Array<IndexExpr> kernel_size;
  for (auto val : param->kernel_size) {
    kernel_size.push_back(val);
  }

  const auto channels = dshape[3];
  const auto out_dtype = param->out_dtype;

  DataType out_dtype_hint = out_dtype;

  // Kernel size in each dimension must be 1, 3, 5, or 7
  for (auto val : kernel_size) {
    // Convert PrimExpr to int
    int ksize = val.as<IntImmNode>()->value;
    CHECK(ksize == 1 || ksize == 3 || ksize == 5 || ksize == 7)
        << "kernel_size should be 1, 3, 5, or 7, but was " << ksize;
  }

  // Check padding mode is SAME or VALID
  const auto padding = param->padding;
  if (padding != "SAME" || padding != "VALID") {
    LOG(FATAL) << "Padding mode " << padding << " is not supported.";
  }

  // Infer output shape from input shape and kernel size and padding mode
  const auto data_layout = param->data_layout;
  Array<IndexExpr> oshape(dshape);

  if (padding == "VALID") {
    // VALID mode means that output size is input size - dilated kernel size + 1
    oshape.Set(2, dshape[2] - param->kernel_size[0] + 1);
    oshape.Set(3, dshape[3] - param->kernel_size[1] + 1);
  }

  // Print output shape
  std::cout << "Output shape: " << Array<IndexExpr>(oshape) << std::endl;

  reporter->Assign(types[1], TensorType(oshape, out_dtype_hint));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.image._make.sobel").set_body_typed(MakeSobel);

RELAY_REGISTER_OP("image.sobel")
    .describe(R"code(Applies the Sobel operator to an image for edge detection.
    - **data**: The input image. Only supports float32 data type. Shape depends on the `data_layout` parameter, if it is `NCHW`,
      the shape is `(batch_size, in_channels, in_height, in_width)`, if `NHWC`, the shape is `(batch_size, in_height, in_width, in_channels)`. 
    - **out**: The output image. Same shape as `data` unless padding mode is `valid` in which case the output shape is `(batch_size, out_height, out_width, in_channels)` where out_height and out_width are computed as:
      out_height = floor((height+2*padding[0]-kernel_size[0])/strides[0])+1
      out_width = floor((width+2*padding[1]-kernel_size[1])/strides[1])+1"
    )code" TVM_ADD_FILELINE)
    .set_attrs_type<SobelAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("Sobel", SobelRel<SobelAttrs>);

// GaussianBlur operator
TVM_REGISTER_NODE_TYPE(GaussianBlurAttrs);

template <typename T>
InferCorrectLayoutOutput InferGaussianBlurOutput(const Attrs& attrs, const Array<Type>& types,
                                                 const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const auto& dshape = data->shape;
  const auto* param = attrs.as<T>();
  CHECK(param != nullptr);
  return InferCorrectLayoutOutput(dshape, param->data_layout);
}

Expr MakeGaussianBlur(Expr data, double sigma, int channels, std::string padding,
                      std::string data_layout, Array<IndexExpr> kernel_size, DataType out_dtype) {
  auto attrs = make_object<GaussianBlurAttrs>();
  attrs->sigma = std::move(sigma);
  attrs->channels = std::move(channels);
  attrs->padding = std::move(padding);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_size = std::move(kernel_size);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("image.gaussian_blur");
  return Call(op, {data}, Attrs(attrs), {});
}

template <typename AttrType>
bool GaussianBlurRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "gaussian_blur: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  const auto dshape = data->shape;
  const auto* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  Array<IndexExpr> kernel_size;
  for (auto val : param->kernel_size) {
    kernel_size.push_back(val);
  }

  const auto channels = dshape[3];
  const auto out_dtype = param->out_dtype;

  DataType out_dtype_hint = out_dtype;

  // Check layout and channels
  Layout layout(param->data_layout);
  CHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
        layout.Contains(LayoutAxis::Get('C')))
      << "Invalid layout " << layout << ". Valid layout is: NCHW.";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  const auto cidx = layout.IndexOf(LayoutAxis::Get('C'));

  reporter->Assign(types[1], TensorType(dshape, out_dtype_hint));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.image._make.gaussian_blur").set_body_typed(MakeGaussianBlur);

RELAY_REGISTER_OP("image.gaussian_blur")
    .describe(R"code(Applies gaussian blur to an image.
    - **data**: The input image. Only support float32 data type.
    - **out**: The output image.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<GaussianBlurAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("GaussianBlur", GaussianBlurRel<GaussianBlurAttrs>);

TVM_REGISTER_NODE_TYPE(ColorSpaceAttrs);

template <typename T>
InferCorrectLayoutOutput InferColorSpaceOutput(const Attrs& attrs, const Array<Type>& types,
                                               const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const auto& dshape = data->shape;
  const auto* param = attrs.as<T>();
  CHECK(param != nullptr);
  return InferCorrectLayoutOutput(dshape, param->data_layout);
}

Expr MakeColorSpace(Expr data, std::string in_mode, std::string out_mode, DataType out_dtype,
                    std::string data_layout) {
  auto attrs = make_object<ColorSpaceAttrs>();
  attrs->out_dtype = std::move(out_dtype);
  attrs->in_mode = std::move(in_mode);
  attrs->out_mode = std::move(out_mode);
  attrs->data_layout = std::move(data_layout);
  static const Op& op = Op::Get("image.color_space");
  return Call(op, {data}, Attrs(attrs), {});
}

template <typename T>
bool ColorSpaceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "color_space: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  const auto dshape = data->shape;
  const auto* param = attrs.as<T>();
  CHECK(param != nullptr);

  static const std::vector<std::string> valid_color_space = {"RGB", "BGR", "GRAY", "HSV", "YUV"};
  CHECK(std::find(valid_color_space.begin(), valid_color_space.end(), param->in_mode) !=
        valid_color_space.end())
      << "Invalid color space " << param->in_mode;
  CHECK(std::find(valid_color_space.begin(), valid_color_space.end(), param->out_mode) !=
        valid_color_space.end())
      << "Invalid color space " << param->out_mode;

  // Check layout and channels
  Layout layout(param->data_layout);
  CHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
        layout.Contains(LayoutAxis::Get('C')))
      << "Invalid layout " << layout << ". Valid layout is: NCHW.";

  const auto& num_channels_expr = dshape[layout.IndexOf(LayoutAxis::Get('C'))];
  const int num_channels = num_channels_expr.as<IntImmNode>()->value;
  // Make sure the number of channels matches the input mode
  if (param->in_mode == "RGB") {
    CHECK_EQ(num_channels, 3) << "Input image must have 3 channels for RGB mode.";
  } else if (param->in_mode == "BGR") {
    CHECK_EQ(num_channels, 3) << "Input image must have 3 channels for BGR mode.";
  } else if (param->in_mode == "GRAY") {
    CHECK_EQ(num_channels, 1) << "Input image must have 1 channel for GRAY mode.";
  } else if (param->in_mode == "HSV") {
    CHECK_EQ(num_channels, 3) << "Input image must have 3 channels for HSV mode.";
  } else if (param->in_mode == "YUV") {
    CHECK_EQ(num_channels, 3) << "Input image must have 3 channels for YUV mode.";
  }
  reporter->Assign(types[1], TensorType(dshape, param->out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.image._make.color_space").set_body_typed(MakeColorSpace);

RELAY_REGISTER_OP("image.color_space")
    .describe(
        R"code(Convert between various color spaces such as BGR, HSV, YUV, GRAYSCALE and RGB.
    The input tensor must be of shape (H, W, C) where H and W are the height and width of the image, and C is the number of color channels. 
    The output tensor has the same shape. 
    The data type of the input tensor is inferred from the input tensor if possible.
    Currently, only float32 is supported.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<ColorSpaceAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("ColorSpace", ColorSpaceRel<ColorSpaceAttrs>);

}  // namespace relay
}  // namespace tvm
