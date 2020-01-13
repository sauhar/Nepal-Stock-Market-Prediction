from django.db import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from datetime import timedelta
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
import tensorflow as tf
#import datafromdatabase
import psycopg2
import warnings
# Create your models here.

class fromdatabase:
    
    def getfromdatabase(company):
        conn= psycopg2.connect(host="localhost",dbname="postgres",user="postgres",password="sauhar" )
        cur = conn.cursor()
        cur.execute("""select openprice,maxprice,minprice,closingprice,date from stockdata where  symbol = '%s' order by date;""" %company)
        row=cur.fetchall()
        conn.commit()
        cur.close()
        #print(row)
        return row
    
    
    
class prediction:
    
    def execute():
        warnings.filterwarnings('ignore')
        
        # Reading the historical data of stocks from the web
        data = fromdatabase.getfromdatabase('ADBL')
        alphabet = "open high low close Date "
        columns = alphabet.split() #split string into a list
        html_data = pd.DataFrame(data,columns=columns) # load the dataset as a pandas data frame
        df = html_data .copy()
        
        # df.head()
        tf.reset_default_graph()
        df.drop(['Date'], 1, inplace = True) # Dropping unnecessary columns
        #print(df)
        plt.figure(figsize=(15, 5));
        plt.plot(df.open.values, color='red', label='open')
        plt.plot(df.close.values, color='green', label='close')
        plt.plot(df.low.values, color='blue', label='low')
        plt.plot(df.high.values, color='black', label='high')
        plt.title('Stock price')
        plt.xlabel('time [days]')
        plt.ylabel('Price in rs')
        plt.legend(loc='best')
        plt.show()
        
        
        
        sc = MinMaxScaler()
        scaled_data = sc.fit_transform(df)
        
        
        tstep = 30
        # since we are looking 60 timesteps back, we can start start looping over only after 60th record in our training set
        data = []
        
        # create all possible sequences of length seq_len
        for i in range(len(scaled_data) - tstep): 
            data.append(scaled_data[i: i + tstep])
        
        data = np.array(data);
        
        # Using 10% of data each for validation and test purpose
        valid_set_size = int(np.round(0.1 * data.shape[0])) 
        test_set_size = valid_set_size
        train_set_size = data.shape[0] - 2*valid_set_size
        # Creating Train data
        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]
        # Creating Validation data
        x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
        y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
        # Creating Test data
        x_test = data[train_set_size+valid_set_size:,:-1,:]
        y_test = data[train_set_size+valid_set_size:,-1,:]
        
        index_in_epoch = 0;
        perm_array  = np.arange(x_train.shape[0])
        np.random.shuffle(perm_array)
        
        # function to get the next batch
        def next_batch(index_in_epoch,batch_size,x_train,perm_array):
           # global index_in_epoch, x_train, perm_array   
            start = index_in_epoch
            index_in_epoch += batch_size
            #print(index_in_epoch)
            if index_in_epoch > x_train.shape[0]:
                #print( x_train.shape[0])
                np.random.shuffle(perm_array) # shuffle permutation array
                start = 0 # start next epoch
                index_in_epoch = batch_size
                
            end = index_in_epoch
            return x_train[perm_array[start:end]], y_train[perm_array[start:end]]
        
        # 4 features
        num_inputs = 4
        # Num of steps in each batch
        num_time_steps = tstep - 1 
        # 100 neuron layer
        num_neurons = 200
        num_outputs = 4
        learning_rate = 0.001 
        # how many iterations to go through (training steps)
        num_train_iterations = 25
        # Size of the batch of data
        batch_size = 50
        # number of LSTM layers
        n_layers = 2
        
        
        # Creating Placeholders for X and y. 
        # The shape for these placeholders should be [None,num_time_steps-1,num_inputs] and [None, num_time_steps-1, num_outputs]
        # The reason we use num_time_steps-1 is because each of these will be one step shorter than the original time steps size, 
        # because we are training the RNN network to predict one point into the future based on the input sequence.
        X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
        y = tf.placeholder(tf.float32, [None, num_outputs])
        
        # use Basic RNN Cell
        cell = [tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.elu)
                for layer in range(n_layers)]
        
        # Creatinmg stacked LSTM
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cell)
        
        # Now pass in the cells variable into tf.nn.dynamic_rnn, along with your first placeholder (X)
        outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32)
        
        stacked_rnn_outputs = tf.reshape(outputs, [-1, num_neurons]) 
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_outputs)
        final_outputs = tf.reshape(stacked_outputs, [-1, num_time_steps, num_outputs])
        final_outputs = final_outputs[:,num_time_steps-1,:] # keep only last output of sequence
        
        # Create a Mean Squared Error Loss Function and use it to minimize an AdamOptimizer.
        loss = tf.reduce_mean(tf.square(final_outputs - y)) # MSE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)
        
        # Initializing the global variable
        init = tf.global_variables_initializer()
        train_set_size = x_train.shape[0]
        
        test_set_size = x_test.shape[0]
        
        saver = tf.train.Saver()
        
        
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session() as sess:
            sess.run(init)    
            for iteration in range(int(num_train_iterations*train_set_size/batch_size)):
                
                x_batch, y_batch = next_batch(index_in_epoch,batch_size,x_train,perm_array)
                sess.run(train, feed_dict={X: x_batch, y: y_batch})
                
                if iteration % 100 == 0:            
                    mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
                    mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
                    print(iteration, '\tTrain MSE:', mse_train, '\tValidation MSE:', mse_valid)
            
            # Saving Model for future use
            saver.save(sess, './model/Stock_prediction_model')
        
        
        with tf.Session() as sess:    
            # Using Saver instance to restore saved rnn 
            saver.restore(sess, './model/Stock_prediction_model')
        
            y_pred = sess.run(final_outputs, feed_dict={X: x_test})
        
        y_test = sc.inverse_transform(y_test)
        y_pred = sc.inverse_transform(y_pred)
        
        # Comparing the actual versus predicted price
        latest_date = max(pd.to_datetime(html_data['Date']))
        ind = []
        predictedtomorrow= y_pred[-1,0]
        print(predictedtomorrow)
         
        for i in range(test_set_size):
            ind.append(latest_date - timedelta(days = test_set_size - i - 1))
        
        fig, ax = plt.subplots(figsize=(15,7))
        plt.plot(ind, y_test[:, 0], color = 'black', label = 'Actual Price') # Plotting the Open Market Price. Hence index 0
        # 0 = open, 1 = close, 2 = highest, 3 = lowest
        ax.plot(ind, y_pred[:, 0], color = 'green', label = 'Predicted Price')
        ax.set_title('Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price in rs')
        # set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        # set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_tick_params(rotation=45)
        ax.legend(loc='best')
        
        plt.savefig('D:/nsap/home/static/home/test.jpeg', format='jpeg')
        
        '''
        plt.show()
        
        # Evaluating the model
        rmse = sqrt(mean_squared_error(y_pred[:,0], y_test[:,0]))
        normalized_rmse = rmse/(max(y_pred[:,0]) - min(y_pred[:,0]))
        print('Normalized RMSE: ', normalized_rmse)
  '''