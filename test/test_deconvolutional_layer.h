/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_cnn/tiny_cnn.h"

namespace tiny_cnn {

TEST(deconvolutional, setup_tiny) {
    deconvolutional_layer<sigmoid> l(2, 2, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::tiny_cnn);

    EXPECT_EQ(l.get_parallelize(), true);       // if layer can be parallelized
    EXPECT_EQ(l.in_channels(), 3);              // num of input tensors
    EXPECT_EQ(l.out_channels(), 2);             // num of output tensors
    EXPECT_EQ(l.in_data_size(), 4);            // size of input tensors
    EXPECT_EQ(l.out_data_size(), 32);           // size of output tensors
    EXPECT_EQ(l.in_data_shape().size(), 1);     // number of inputs shapes
    EXPECT_EQ(l.out_data_shape().size(), 1);    // num of output shapes
    EXPECT_EQ(l.get_weights().size(), 2);       // the wieghts vector size
    EXPECT_EQ(l.get_weight_grads().size(), 2);  // the wieghts vector size
    EXPECT_EQ(l.get_inputs().size(), 3);        // num of input edges
    EXPECT_EQ(l.get_outputs().size(), 2);       // num of outpus edges
    EXPECT_EQ(l.in_types().size(), 3);          // num of input data types
    EXPECT_EQ(l.out_types().size(), 2);         // num of output data types
    EXPECT_EQ(l.fan_in_size(), 9);              // num of incoming connections
    EXPECT_EQ(l.fan_out_size(), 18);            // num of outgoing connections
    EXPECT_STREQ(l.layer_type().c_str(), "deconv");  // string with layer type
    EXPECT_TRUE(l.get_backend_type() == backend_t::tiny_cnn);
}

#ifdef CNN_USE_NNPACK
TEST(deconvolutional, setup_nnp) {
    deconvolutional_layer<sigmoid> l(2, 2, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::nnpack);

    EXPECT_EQ(l.get_parallelize(), true);       // if layer can be parallelized
    EXPECT_EQ(l.in_channels(), 3);              // num of input tensors
    EXPECT_EQ(l.out_channels(), 2);             // num of output tensors
    EXPECT_EQ(l.in_data_size(), 4);            // size of input tensors
    EXPECT_EQ(l.out_data_size(), 32);           // size of output tensors
    EXPECT_EQ(l.in_data_shape().size(), 1);     // number of inputs shapes
    EXPECT_EQ(l.out_data_shape().size(), 1);    // num of output shapes
    EXPECT_EQ(l.get_weights().size(), 2);       // the wieghts vector size
    EXPECT_EQ(l.get_weight_grads().size(), 2);  // the wieghts vector size
    EXPECT_EQ(l.get_inputs().size(), 3);        // num of input edges
    EXPECT_EQ(l.get_outputs().size(), 2);       // num of outpus edges
    EXPECT_EQ(l.in_types().size(), 3);          // num of input data types
    EXPECT_EQ(l.out_types().size(), 2);         // num of output data types
    EXPECT_EQ(l.fan_in_size(), 9);              // num of incoming connections
    EXPECT_EQ(l.fan_out_size(), 18);            // num of outgoing connections
    EXPECT_STREQ(l.layer_type().c_str(), "deconv");  // string with layer type
    EXPECT_TRUE(l.get_backend_type() == backend_t::nnpack);
}
#endif

TEST(deconvolutional, fprop) {
    typedef network<sequential> CNN;
    CNN nn;

    deconvolutional_layer<sigmoid> l(2, 2, 3, 1, 2);

    // layer::forward_propagation expects tensors, even if we feed only one input at a time
    auto create_simple_tensor = [](size_t vector_size) {
        return tensor_t(1, vec_t(vector_size));
    };

    // create simple tensors that wrap the payload vectors of the correct size
    tensor_t in_tensor     = create_simple_tensor(4)
           , out_tensor    = create_simple_tensor(32)
           , a_tensor      = create_simple_tensor(32)
           , weight_tensor = create_simple_tensor(18)
           , bias_tensor   = create_simple_tensor(2);

    // short-hand references to the payload vectors
    vec_t &in     = in_tensor[0]
        , &out    = out_tensor[0]
        , &weight = weight_tensor[0];

    ASSERT_EQ(l.in_shape()[1].size(), 18); // weight

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    std::vector<tensor_t*> in_data, out_data;
    in_data.push_back(&in_tensor);
    in_data.push_back(&weight_tensor);
    in_data.push_back(&bias_tensor);
    out_data.push_back(&out_tensor);
    out_data.push_back(&a_tensor);
    l.setup(false);
    {
        l.forward_propagation(in_data, out_data);

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, (tiny_cnn::float_t)0.5);
    }

    weight[0] = 0.3;  weight[1] = 0.1; weight[2] = 0.2;
    weight[3] = 0.0;  weight[4] =-0.1; weight[5] =-0.1;
    weight[6] = 0.05; weight[7] =-0.2; weight[8] = 0.05;

    weight[9]  = 0.0; weight[10] =-0.1; weight[11] = 0.1;
    weight[12] = 0.1; weight[13] =-0.2; weight[14] = 0.3;
    weight[15] = 0.2; weight[16] =-0.3; weight[17] = 0.2;

    in[0] = 3;  in[1] = 2;
    in[2] = 3;  in[3] = 0;

    {
        l.forward_propagation(in_data, out_data);

        EXPECT_NEAR(0.7109495, out[0], 1E-5);
        EXPECT_NEAR(0.7109495, out[1], 1E-5);
        EXPECT_NEAR(0.6899745, out[2], 1E-5);
        EXPECT_NEAR(0.5986877, out[3], 1E-5);
        EXPECT_NEAR(0.7109495, out[4], 1E-5);
        EXPECT_NEAR(0.5000000, out[5], 1E-5);
        EXPECT_NEAR(0.5249792, out[6], 1E-5);
        EXPECT_NEAR(0.4501660, out[7], 1E-5);
        EXPECT_NEAR(0.5374298, out[8], 1E-5);
        EXPECT_NEAR(0.3100255, out[9], 1E-5);
        EXPECT_NEAR(0.3658644, out[10], 1E-5);
        EXPECT_NEAR(0.5249791, out[11], 1E-5);
        EXPECT_NEAR(0.5374298, out[12], 1E-5);
        EXPECT_NEAR(0.3543437, out[13], 1E-5);
        EXPECT_NEAR(0.5374298, out[14], 1E-5);
        EXPECT_NEAR(0.5000000, out[15], 1E-5);
    }
}

TEST(deconvolutional, fprop2) {
    typedef network<sequential> CNN;
    CNN nn;

    deconvolutional_layer<sigmoid> l(2, 2, 3, 1, 2, padding::same);

    auto create_simple_tensor = [](size_t vector_size) {
        return tensor_t(1, vec_t(vector_size));
    };

    tensor_t in_tensor     = create_simple_tensor(4)
           , out_tensor    = create_simple_tensor(32)
           , a_tensor      = create_simple_tensor(32)
           , weight_tensor = create_simple_tensor(18)
           , bias_tensor   = create_simple_tensor(2);

    // short-hand references to the payload vectors
    vec_t &in = in_tensor[0]
        , &out = out_tensor[0]
        , &weight = weight_tensor[0];

    ASSERT_EQ(l.in_shape()[1].size(), 18); // weight

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    std::vector<tensor_t*> in_data, out_data;
    in_data.push_back(&in_tensor);
    in_data.push_back(&weight_tensor);
    in_data.push_back(&bias_tensor);
    out_data.push_back(&out_tensor);
    out_data.push_back(&a_tensor);
    l.setup(false);
    {
        l.forward_propagation(in_data, out_data);

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, (tiny_cnn::float_t)0.5);
    }

    weight[0] = 0.3;  weight[1] = 0.1; weight[2] = 0.2;
    weight[3] = 0.0;  weight[4] =-0.1; weight[5] =-0.1;
    weight[6] = 0.05; weight[7] =-0.2; weight[8] = 0.05;

    weight[9]  = 0.0; weight[10] =-0.1; weight[11] = 0.1;
    weight[12] = 0.1; weight[13] =-0.2; weight[14] = 0.3;
    weight[15] = 0.2; weight[16] =-0.3; weight[17] = 0.2;

    in[0] = 3;  in[1] = 2;
    in[2] = 3;  in[3] = 0;

    {
        l.forward_propagation(in_data, out_data);

        EXPECT_NEAR(0.5000000, out[0], 1E-5);
        EXPECT_NEAR(0.5249792, out[1], 1E-5);
        EXPECT_NEAR(0.3100255, out[2], 1E-5);
        EXPECT_NEAR(0.3658644, out[3], 1E-5);
    }
}

TEST(deconvolutional, gradient_check) { // tanh - mse
    network<sequential> nn;
    nn << deconvolutional_layer<tan_h>(2, 2, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check2) { // sigmoid - mse
    network<sequential> nn;
    nn << deconvolutional_layer<sigmoid>(2, 2, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check3) { // rectified - mse
    network<sequential> nn;

    nn << deconvolutional_layer<rectified_linear>(2, 2, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check4) { // identity - mse
    network<sequential> nn;

    nn << deconvolutional_layer<identity>(2, 2, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, gradient_check5) { // sigmoid - cross-entropy
    network<sequential> nn;

    nn << deconvolutional_layer<sigmoid>(2, 2, 3, 1, 1);

    const auto test_data = generate_gradient_check_data(nn.in_data_size());
    nn.init_weight();
    EXPECT_TRUE(nn.gradient_check<cross_entropy>(test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(deconvolutional, read_write)
{
    deconvolutional_layer<tan_h> l1(2, 2, 3, 1, 1);
    deconvolutional_layer<tan_h> l2(2, 2, 3, 1, 1);

    l1.init_weight();
    l2.init_weight();

    serialization_test(l1, l2);
}

TEST(deconvolutional, read_write2) {
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O,
        O, O, X, X, X, O,
        O, O, O, X, X, X
    };
#undef O
#undef X
    deconvolutional_layer<tan_h> layer1(14, 14, 5, 3, 6, connection_table(connection, 3, 6));
    deconvolutional_layer<tan_h> layer2(14, 14, 5, 3, 6, connection_table(connection, 3, 6));
    layer1.init_weight();
    layer2.init_weight();

    serialization_test(layer1, layer2);
}


} // namespace tiny-cnn
