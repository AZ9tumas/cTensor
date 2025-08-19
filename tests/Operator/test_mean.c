#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_mean_operator() {
    const char* op_name = "mean";
    PoolId pool_id = 3;
    cten_begin_malloc(pool_id);

    TensorShape exp_shape_scalar = {1};

    // Test Case 1: Mean of a scalar tensor
    {
        const char* tc_name = "mean_scalar";
        TensorShape s_shape = {1};
        float d1[] = {5.0f};
        float exp_d[] = {5.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Mean of a vector
    {
        const char* tc_name = "mean_vector_1D";
        TensorShape v_shape = {3};
        float d1[] = {1.0f, 2.0f, 3.0f};  // Sum = 6, Count = 3, Mean = 2
        float exp_d[] = {2.0f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Mean of a matrix
    {
        const char* tc_name = "mean_matrix_2x2";
        TensorShape m_shape = {2, 2};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};  // Sum = 10, Count = 4, Mean = 2.5
        float exp_d[] = {2.5f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Mean of a matrix with negative numbers
    {
        const char* tc_name = "mean_matrix_2x2_negative";
        TensorShape m_shape = {2, 2};
        float d1[] = {-1.0f, 2.0f, -3.0f, 4.0f};  // Sum = 2, Count = 4, Mean = 0.5
        float exp_d[] = {0.5f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Mean of a tensor with all zeros
    {
        const char* tc_name = "mean_vector_all_zeros";
        TensorShape v_shape = {4};
        float d1[] = {0.0f, 0.0f, 0.0f, 0.0f};  // Sum = 0, Count = 4, Mean = 0
        float exp_d[] = {0.0f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Large Tensor Reductions
    {
        const char* tc_name = "mean_large_tensor_reductions";

        // Sub-test 1: Large tensor mean (1,000 elements)
        {
            TensorShape large_shape = {1000};
            float large_data[1000];
            for(int i = 0; i < 1000; i++)
                large_data[i] = 1.0f;

            float exp_d[] = {1.0f};  // Mean of 1000 ones = 1.0
            TensorShape exp_shape = {1};

            Tensor t1 = create_test_tensor(large_shape, large_data, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Large count division (stress test - 5,000 elements)
        {
            TensorShape stress_shape = {5000};
            float stress_data[5000];
            for(int i = 0; i < 5000; i++)
                stress_data[i] = 2.0f;

            float exp_d[] = {2.0f};  // Mean of 5000 twos = 2.0
            TensorShape exp_shape = {1};

            Tensor t1 = create_test_tensor(stress_shape, stress_data, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Higher Dimensional Tensors
    {
        const char* tc_name = "mean_higher_dimensional_tensors";

        // Sub-test 1: 3D tensor mean along axis 1 (3x4x5 -> 3x5)
        {
            TensorShape shape_3d = {3, 4, 5};
            float d1[] = {0.0761f, 0.8512f, 0.4951f, 0.4806f, 0.5924f, 0.8247f, 0.3478f, 0.6780f,
                          0.5657f, 0.2670f, 0.8786f, 0.7974f, 0.6585f, 0.8506f, 0.8673f, 0.7084f,
                          0.8370f, 0.6975f, 0.6801f, 0.6186f, 0.7527f, 0.1586f, 0.8809f, 0.8718f,
                          0.0292f, 0.8258f, 0.1289f, 0.3351f, 0.7435f, 0.1608f, 0.8180f, 0.8321f,
                          0.5075f, 0.0064f, 0.2870f, 0.6169f, 0.9812f, 0.6318f, 0.2598f, 0.6340f,
                          0.5400f, 0.7798f, 0.1070f, 0.7610f, 0.5413f, 0.9630f, 0.3419f, 0.6326f,
                          0.9320f, 0.1025f, 0.9372f, 0.6879f, 0.0678f, 0.3010f, 0.7082f, 0.0674f,
                          0.5822f, 0.3459f, 0.6209f, 0.0457f};
            TensorShape exp_shape = {3, 5};
            float exp_d[] = {0.6220f,
                             0.7084f,
                             0.6323f,
                             0.6443f,
                             0.5863f,
                             0.7534f,
                             0.5252f,
                             0.5888f,
                             0.4704f,
                             0.2778f,
                             0.6269f,
                             0.5979f,
                             0.2883f,
                             0.6537f,
                             0.3494f};

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1, 1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor mean along axis 2 (2x3x4x5 -> 2x3x5)
        // Testing negative index case so 2 becomes -2
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[] = {0.8715f, 0.9735f, 0.9689f, 0.7497f, 0.1301f, 0.7583f, 0.0246f, 0.0221f,
                          0.3236f, 0.4886f, 0.7704f, 0.6833f, 0.4459f, 0.2736f, 0.9971f, 0.4262f,
                          0.4514f, 0.1636f, 0.7948f, 0.6937f, 0.2208f, 0.0824f, 0.6805f, 0.6545f,
                          0.2733f, 0.9509f, 0.1511f, 0.4323f, 0.9436f, 0.4197f, 0.6385f, 0.3976f,
                          0.2742f, 0.9840f, 0.4093f, 0.8941f, 0.2300f, 0.2131f, 0.0311f, 0.6517f,
                          0.3685f, 0.8644f, 0.4732f, 0.9682f, 0.1855f, 0.8686f, 0.7766f, 0.7709f,
                          0.8448f, 0.7610f, 0.6262f, 0.1312f, 0.0325f, 0.9208f, 0.6167f, 0.7965f,
                          0.4815f, 0.1173f, 0.1252f, 0.6856f, 0.4303f, 0.2005f, 0.4916f, 0.0642f,
                          0.5820f, 0.2690f, 0.7976f, 0.3104f, 0.4552f, 0.0116f, 0.0724f, 0.3925f,
                          0.4799f, 0.6000f, 0.2917f, 0.6950f, 0.8601f, 0.7799f, 0.0396f, 0.4805f,
                          0.1049f, 0.2420f, 0.9867f, 0.1425f, 0.4989f, 0.6182f, 0.7025f, 0.5596f,
                          0.0098f, 0.3265f, 0.5177f, 0.0879f, 0.3506f, 0.0332f, 0.0786f, 0.3969f,
                          0.1327f, 0.5675f, 0.6895f, 0.8006f, 0.2002f, 0.1675f, 0.1046f, 0.6364f,
                          0.7065f, 0.0316f, 0.9362f, 0.0520f, 0.5413f, 0.7091f, 0.8710f, 0.7141f,
                          0.8017f, 0.3395f, 0.8148f, 0.0801f, 0.8948f, 0.5476f, 0.8173f, 0.4523f};
            TensorShape exp_shape = {2, 3, 5};
            float exp_d[] = {0.7066f, 0.5332f, 0.4001f, 0.5354f, 0.5774f, 0.6761f, 0.2152f, 0.4000f,
                             0.6533f, 0.4385f, 0.6650f, 0.5634f, 0.3485f, 0.7148f, 0.5622f, 0.3667f,
                             0.5627f, 0.5154f, 0.2898f, 0.3414f, 0.4094f, 0.2913f, 0.6161f, 0.2187f,
                             0.4261f, 0.2957f, 0.6781f, 0.3765f, 0.5836f, 0.6707f};

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1, -2);  // Testing negative index case so 2 becomes -2

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
        // Sub-test 3: 4D tensor mean along axis 1 and axis 2 (2x3x4x5 -> 2x5)
        // Here I want to test the negative index case, Here first we do mean along axis 2 (which is
        // -2) Now we become (2x3x5) and then we do mean along axis 1 (which is -2) again Now we
        // become (2x5)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[120] = {
                0.374540f, 0.950714f, 0.731994f, 0.598658f, 0.156019f, 0.155995f, 0.058084f,
                0.866176f, 0.601115f, 0.708073f, 0.020584f, 0.969910f, 0.832443f, 0.212339f,
                0.181825f, 0.183405f, 0.304242f, 0.524756f, 0.431945f, 0.291229f, 0.611853f,
                0.139494f, 0.292145f, 0.366362f, 0.456070f, 0.785176f, 0.199674f, 0.514234f,
                0.592415f, 0.046450f, 0.607545f, 0.170524f, 0.065052f, 0.948886f, 0.965632f,
                0.808397f, 0.304614f, 0.097672f, 0.684233f, 0.440152f, 0.122038f, 0.495177f,
                0.034389f, 0.909320f, 0.258780f, 0.662522f, 0.311711f, 0.520068f, 0.546710f,
                0.184854f, 0.969585f, 0.775133f, 0.939499f, 0.894827f, 0.597900f, 0.921874f,
                0.088493f, 0.195983f, 0.045227f, 0.325330f, 0.388677f, 0.271349f, 0.828738f,
                0.356753f, 0.280935f, 0.542696f, 0.140924f, 0.802197f, 0.074551f, 0.986887f,
                0.772245f, 0.198716f, 0.005522f, 0.815461f, 0.706857f, 0.729007f, 0.771270f,
                0.074045f, 0.358466f, 0.115869f, 0.863103f, 0.623298f, 0.330898f, 0.063558f,
                0.310982f, 0.325183f, 0.729606f, 0.637557f, 0.887213f, 0.472215f, 0.119594f,
                0.713245f, 0.760785f, 0.561277f, 0.770967f, 0.493796f, 0.522733f, 0.427541f,
                0.025419f, 0.107891f, 0.031429f, 0.636410f, 0.314356f, 0.508571f, 0.907566f,
                0.249292f, 0.410383f, 0.755551f, 0.228798f, 0.076980f, 0.289751f, 0.161221f,
                0.929698f, 0.808120f, 0.633404f, 0.871461f, 0.803672f, 0.186570f, 0.892559f,
                0.539342f};
            TensorShape exp_shape = {2, 5};
            float exp_d[10] = {0.518626f,
                               0.397314f,
                               0.467868f,
                               0.569336f,
                               0.384360f,
                               0.473020f,
                               0.498569f,
                               0.504455f,
                               0.465062f,
                               0.492491f};

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor temp_res = Tensor_mean(t1, -2);
            Tensor actual_res = Tensor_mean(temp_res, -2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 8: Edge Cases
    {
        const char* tc_name = "mean_edge_cases";

        // Sub-test 1: Single element mean (division by 1)
        {
            TensorShape single_shape = {1};
            float d1[] = {42.5f};
            float exp_d[] = {42.5f};  // Mean of single element is itself

            Tensor t1 = create_test_tensor(single_shape, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
