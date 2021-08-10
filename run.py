import load_dataset,implementation

training_data, validation_data, test_data = load_dataset.load_dataset()
neural_net = implementation.Model([784,30,10])
neural_net.SGD(training_data, epochs=28, mini_batch_size=10, eta=3.0,test_data=test_data)
