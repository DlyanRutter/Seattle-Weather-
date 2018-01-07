import pandas as pd
import numpy as np
import keras, math
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

data_dir = '/Users/dylanrutter/Downloads/seattleweather.csv'
save_file = '/tmp/model.ckpt'
weight_file = 'mine'

def get_data():
    """
    Retrieves the dataframe from source location. Replaces "DATE" category
    with either 0, indicating the date was during the spring season, 1,
    indicating summer, 2, indicating fall, 3, indicating winter, or 4,
    indicating an invalid date or no date was entered. Also replaces the
    boolean "RAIN" column with floating point 0.0 or 1.0. Returns a
    numpy array for features, a numpy array for labels, and a
    one hot array for labels
    """
    df = pd.read_csv(data_dir)
    df.dropna()
    date = np.array((df.pop('DATE')))
    season = []

    spring = ['03', '04', '05']
    summer = ['06', '07', '08']
    fall = ['09', '10', '11']
    winter = ['12', '01', '02']
    
    for e in date:
        month = e[5:7]

        if month in spring: season.append(0)
        elif month in summer: season.append(1)
        elif month in fall: season.append(2)
        elif month in winter: season.append(3)
        else: season.append(4)
    
    season = pd.Series(season)
    df["SEASON"] = season

    labels = np.asarray(df['RAIN'], dtype="|S6")
    one_hot = LabelBinarizer()
    df.pop('RAIN')
     
    return df.as_matrix(), labels, one_hot.fit_transform(labels)

def accuracy(predictions, labels):
    """
    determines accuracy
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


features, labels, one_hot = get_data()
features = features.astype(np.float32)
X_train,X_valid,X_test = features[:20000],features[20001:22000],features[22001:]
y_train, y_valid, y_test = one_hot[:20000],one_hot[20001:22000],one_hot[22001:]

num_features = X_train.shape[1]
num_labels = y_train.shape[1]
batch_size = 20
epochs = 3000
learn_rate = 0.01
num_steps = 3001

#number of nodes in each hidden layer
h1_nodes = 528
h2_nodes = 256
h3_nodes = 128

graph = tf.Graph()
with graph.as_default():

    #make placeholders for batches
    tf_train_features = tf.placeholder(tf.float32,
                                       shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_features = tf.constant(X_valid)
    tf_test_features = tf.constant(X_test)

    #set up weights and biases
    initializer = tf.contrib.layers.xavier_initializer()
    hl1 = {'weights1': tf.Variable(initializer([num_features, h1_nodes])),
           'biases1' : tf.Variable(tf.zeros([h1_nodes]))}
    hl2 = {'weights2': tf.Variable(initializer([h1_nodes, h2_nodes])),
           'biases2' : tf.Variable(tf.zeros([h2_nodes]))}
    hl3 = {'weights3': tf.Variable(initializer([h2_nodes, h3_nodes])),
           'biases3' : tf.Variable(tf.zeros([h3_nodes]))}
    out = {'weights4': tf.Variable(initializer([h3_nodes, num_labels])),
           'biases4' : tf.Variable(tf.zeros([num_labels]))}
    """
    #save weights    
    weight_saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "Weights: "
        print sess.run(hl1['weights1'])
        print sess.run(hl2['weights2'])
        print sess.run(hl3['weights3'])
        print sess.run(out['weights4'])
        print "Biases: "
        print sess.run(hl1['biases1'])
        print sess.run(hl2['biases2'])
        print sess.run(hl3['biases3'])
        print sess.run(out['biases4'])
        weight_saver.save(sess, 'my model')
    """                       
    #establish a keep_probability placeholder for dropout and compute logits
    #for each layer
    keep_prob = tf.placeholder(tf.float32)
    lg1 = tf.add(tf.matmul(tf_train_features, hl1['weights1']), hl1['biases1'])
    lg1 = tf.nn.relu(lg1)
    lg1 = tf.nn.dropout(lg1, keep_prob)

    lg2 = tf.add(tf.matmul(lg1, hl2["weights2"]), hl2['biases2'])
    lg2 = tf.nn.relu(lg2)
    lg2 = tf.nn.dropout(lg2, keep_prob)

    lg3 = tf.add(tf.matmul(lg2, hl3['weights3']), hl3['biases3'])
    lg3 = tf.nn.relu(lg3)
    lg3 = tf.nn.dropout(lg3, keep_prob)
    logits = tf.matmul(lg3, out['weights4']) + out['biases4']
    
    #compute loss function of cross entropy + regularization
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf_train_labels)) +\
        0.01*tf.nn.l2_loss(hl1['weights1']) +\
        0.01* tf.nn.l2_loss(hl2['weights2']) + 0.01*tf.nn.l2_loss(
            hl3['weights3']) + 0.01*tf.nn.l2_loss(out['weights4'])

    #make an optimizer with a decaying learning rate               
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 10000, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
    
    #make logits for validation set
    Vlg1 = tf.add(tf.matmul(tf_valid_features, hl1['weights1']), hl1['biases1'])
    Vlg1 = tf.nn.relu(Vlg1)

    Vlg2 = tf.add(tf.matmul(Vlg1, hl2["weights2"]), hl2['biases2'])
    Vlg2 = tf.nn.relu(Vlg2)

    Vlg3 = tf.add(tf.matmul(Vlg2, hl3['weights3']), hl3['biases3'])
    Vlg3 = tf.nn.relu(Vlg3)
    Vlogits = tf.matmul(Vlg3, out['weights4']) + out['biases4']
    
    #make logits for test set
    Tlg1 = tf.add(tf.matmul(tf_test_features, hl1['weights1']), hl1['biases1'])
    Tlg1 = tf.nn.relu(Tlg1)

    Tlg2 = tf.add(tf.matmul(Tlg1, hl2["weights2"]), hl2['biases2'])
    Tlg2 = tf.nn.relu(Tlg2)

    Tlg3 = tf.add(tf.matmul(Tlg2, hl3['weights3']), hl3['biases3'])
    Tlg3 = tf.nn.relu(Tlg3)
    Tlogits = tf.matmul(Tlg3, out['weights4']) + out['biases4']
    
    #set up predictions/create probabilities
    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(Vlogits)
    test_pred = tf.nn.softmax(Tlogits)

#initialize session
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    #run step number of epochs
    for step in range(epochs):

        #establish batches
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_features = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]

        #create feed dict mapping placeholders to batches, learning rate to
        #learning rate, and keep probability to keep probability
        train_feed_dict = {tf_train_features:batch_features,
                           tf_train_labels:batch_labels,
                           learning_rate:learn_rate,
                           keep_prob:0.5}

        #run session to find log loss and predictions
        _, l, predictions = sess.run(
            [optimizer, loss, train_pred], feed_dict=train_feed_dict)
        
        #Get stats
        if (step % 500 == 0):
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(
                accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(
                accuracy(valid_pred.eval(), y_valid)))
    print("Test accuracy: {:.1f}".format(accuracy(test_pred.eval(), y_test)))

    #save model/session
   # saver.save(sess, save_file)
    print "Trained model saved!"

