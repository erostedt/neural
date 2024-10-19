# About
neural is a neural network library written in pure C with no only C standard library and math as dependencies.

# Pros
* Simple syntax
* Easy to build and setup
* No external dependencies
* Minimal memory allocations
* Supports most common activation function (ReLU, Sigmoid, Tanh, Softmax and Linear)
* Supports most common loss functions (MSE, Binary crossentropy and categorical crossentropy)
* Uses adam optimizer

# Cons
* Cpu only
* Only dense layer
* No batch normalization or Dropout

# How to use
See examples. But generally something like:
```
    const size_t BATCH_SIZE = 16;
    const size_t OUTPUT_SIZE = 3;
    const double TRAINING_FRACTION = 0.7;
    const double LEARNING_RATE = 1e-3;
    const size_t EPOCHS = 10000;
    const size_t SEED = 39;
    const loss_type_t LOSS = CATEGORICAL_CROSS_ENTROPY;

    set_seed(SEED);
    matrix_t features = ...;
    matrix_t targets = ...;

    dataset_t dataset = train_test_split(features, targets, TRAINING_FRACTION);
    // Optional
    standardization_t standardization = calculate_standardization(dataset.train_features);
    standardize(dataset.train_features, dataset.train_features, standardization);
    standardize(dataset.test_features, dataset.test_features, standardization);

    layer_type_t layers[] = {
        LAYER_RELU(8),
        LAYER_RELU(16),
        LAYER_SOFTMAX(OUTPUT_SIZE),
    };
    network_t network = network_alloc(BATCH_SIZE, features.cols, layers, ARRAY_LEN(layers), LOSS);
    network_summary(&network);

    adam_parameters_t optimizer = optimizer_default(LEARNING_RATE);
    network_train(&network, dataset.train_features, dataset.train_targets, optimizer, EPOCHS);

    matrix_t pred = matrix_alloc_like(dataset.test_targets);
    network_predict(&network, dataset.test_features, pred);

```
