from RNNSupport import *

def dnn2_rnn3(length_of_sequences, hidden_neurons, loss, optimizer, activation, rate, dropout, batch_size = None, stateful = False):
    in_out_neurons = 1
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons)) 
    dnn = Dense(hidden_neurons/2, activation=activation, name='dnn')(inp)
    dnn1 = Dense(hidden_neurons/2, activation=activation, name='dnn1')(dnn)
    rnn1 = SimpleRNN(hidden_neurons, 
                    return_sequences=True,
                    stateful = stateful,
                    name="RNN1", use_bias=True,recurrent_dropout=dropout, kernel_regularizer=keras.regularizers.l2(rate))(dnn1)
    rnn2 = SimpleRNN(hidden_neurons, 
                    return_sequences=True,
                    stateful = stateful,
                    name="RNN2", use_bias=True,recurrent_dropout=dropout, kernel_regularizer=keras.regularizers.l2(rate))(rnn1)
    rnn = SimpleRNN(hidden_neurons, 
                    return_sequences=False,
                    stateful = stateful,
                    name="RNN", use_bias=True,recurrent_dropout=dropout, kernel_regularizer=keras.regularizers.l2(rate))(rnn2)
    dens = Dense(in_out_neurons,name="dense")(rnn)
    model = Model(inputs=[inp],outputs=[dens])
    model.compile(loss=loss, optimizer=optimizer)  
    return model


# Vary Dimension
#datatype='VaryDimension'
#X_tot = np.arange(2, 42, 2)
y_tot = np.array([-0.03077640549, -0.08336233266, -0.1446729567, -0.2116753732, -0.2830637392, -0.3581341341, -0.436462435, -0.5177783846,
	-0.6019067271, -0.6887363571, -0.7782028952, -0.8702784034, -0.9649652536, -1.062292565, -1.16231451, 
	-1.265109911, -1.370782966, -1.479465113, -1.591317992, -1.70653767])

dim=12
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

    for num in range(10, 540, 40):
        for act in [None, 'tanh', 'relu', 'sigmoid']:
            for rate in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
                for dropout in [0.0, 0.05, 0.1, 0.2, 0.25, 0.5]:
                    errors = []
                    for i in range (5):
                        print('***********', seq, num, act, rate, dropout)
                        model = dnn2_rnn3(length_of_sequences = seq, hidden_neurons = num, loss=loss, optimizer = optimizer, activation=act, rate=rate, dropout=dropout)
                        iterations = 200
                        model.fit (X_train, y_train, epochs=iterations, validation_split=0.0, verbose=False)

                        y_return = []

                        y_return = y_tot[:dim].tolist()
                        next_input = np.array([[[y_return[-2]], [y_return[-1]]]])
                        last = [y_return[-1]]

                        total_points = 20

                        while len(y_return) < total_points:
                            next = model.predict(next_input)
                            y_return.append(next[0][0])
                            next_input = np.array([[last, next[0]]])
                            last = next[0]
                        mse_err = mse(y_return, y_tot)
                        if mse_err < best_score:
                            best_score = mse_err
                            best_model = [seq, num, act, rate, dropout]
                            best_extrapolation = y_return
                            print("BEST SCORE: ", best_score)
                            print("BEST PARAMETERS: ", best_model)
                            print("BEST EXTRAPOLATION: ", best_extrapolation)
                        if mse_err > worst_score:
                            worst_score = mse_err
                            worst_model = [seq, num, act, rate, dropout]
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



