from RNNSupport import *

def rnn(seq, hidden_neurons):
    """
        Inputs:
            length_of_sequences (an int): the length of sequence used to format the data set
            hidden_neurons (an int): the number of neurons to be used in the SimpleRNN layers, or double the number
                of neurons to be used in the Dense layers
            loss (a string): the loss function to be used
            optimizer (a string): the optimizer to be used
            activation (a string): the activation to be used in the dense layers
            rate (an int or float): the L2 regulization rate (not used in this example)
            batch_size (an int): Default value is None.  See Keras documentation of SimpleRNN.
            stateful (a boolean): Default value is False.  See Keras documentation of SimpleRNN.
        Returns:
            model (a Keras model): a compiled recurrent neural network consisting of one input layer followed by
                2 dense (feedforward layers), then three simple recurrent neural network layers, and finally an
                output layer.
        Builds and compiles a Keras recurrent neural network with specified parameters using two hidden dense 
        layers followed by three hidden simple recurrent neural network layers.
    """
    rate = 0
    # Number of neurons in the input and output layers
    in_out_neurons = 1
    activation = 'relu'
    # Input layer
    inp = Input(batch_shape=(None, 
                seq, 
                in_out_neurons)) 
    # Hidden dense layers
    #dnn = Dense(hidden_neurons/2, activation=activation, name='dnn')(inp)
    #dnn1 = Dense(hidden_neurons/2, activation=activation, name='dnn1')(dnn)
    # Hidden simple recurrent layers
    rnn1 = SimpleRNN(hidden_neurons, 
                    return_sequences=True,
                    stateful = False,
                    name="RNN1", use_bias=True,recurrent_dropout=0.0, kernel_regularizer=keras.regularizers.l2(rate))(inp)
    rnn2 = SimpleRNN(hidden_neurons, 
                    return_sequences=True,
                    stateful = False,
                    name="RNN2", use_bias=True,recurrent_dropout=0.0, kernel_regularizer=keras.regularizers.l2(rate))(rnn1)
    rnn = SimpleRNN(hidden_neurons, 
                    return_sequences=False,
                    stateful = False,
                    name="RNN", use_bias=True,recurrent_dropout=0.0, kernel_regularizer=keras.regularizers.l2(rate))(rnn2)
    # Output layer
    dens = Dense(in_out_neurons,name="dense")(rnn)
    # Build the model
    model = Model(inputs=[inp],outputs=[dens])
    # Compile the model
    model.compile(loss='mse', optimizer='adam')  
    # Return the model
    return model


# Vary Dimension
#datatype='VaryDimension'
#X_tot = np.arange(2, 42, 2)
y_tot = np.arange(0, 8*np.pi, 0.1)

dim=int(len(y_tot)/2)
y_train = y_tot[:dim]


best_score = 100
best_model = []
best_extrapolation = []

worst_score = 0
worst_model = []
worst_extrapolation = []

total_errors = []

for seq in [2]:
    X_train, y_train = format_data (y_train, seq)

    loss = 'mse'
    optimizer = 'adam'

    for neuron in range(10, 540, 40):
        for act in ['tanh']:
            for rate in [0]:
                for dropout in [0.0]:
                    for epoch in range(50, 1050, 50):
                        errors = []
                        for i in range (5):
                            print('***********', seq, neuron, act, rate, dropout, epoch)
                            model = rnn(2, neuron)
                            #iterations = 200
                            model.fit (X_train, y_train, epochs=epoch, validation_split=0.0, verbose=False)

                            y_return = []

                            y_return = y_tot[:dim].tolist()
                            next_input = np.array([[[y_return[-2]], [y_return[-1]]]])
                            last = [y_return[-1]]

                            total_points = len(y_tot)

                            while len(y_return) < total_points:
                                next = model.predict(next_input)
                                y_return.append(next[0][0])
                                next_input = np.array([[last, next[0]]])
                                last = next[0]
                            mse_err = mse(y_return, y_tot)
                            if mse_err < best_score:
                                best_score = mse_err
                                best_model = [neuron, epoch]
                                best_extrapolation = y_return
                                print("BEST SCORE: ", best_score)
                                print("BEST PARAMETERS: ", best_model)
                                print("BEST EXTRAPOLATION: ", best_extrapolation)
                            if mse_err > worst_score:
                                worst_score = mse_err
                                worst_model = [neuron, epoch]
                                worst_extrapolation = y_return   
                                print("WORST SCORE: ", worst_score)
                                print("WORSTPARAMETERS: ", worst_model)
                                print("WORST EXTRAPOLATION: ", worst_extrapolation)                         

                            errors.append (mse_err)
                            print()
                        total_errors.append(np.average(errors))




print("RESULTS")
print("BEST SCORE: ", best_score)
print("BEST PARAMETERS: ", best_model)
print("BEST EXTRAPOLATION: ", best_extrapolation)
print()
print("WORST SCORE: ", worst_score)
print("WORSTPARAMETERS: ", worst_model)
print("WORST EXTRAPOLATION: ", worst_extrapolation)
print()
print("TOTAL ERRORS: ", total_errors)



