import tensorflow as tf
import random
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)
# one_hot = True, 4-> 0, 0, 0, 0, 1, 0, 0, 0, 0, 0

X = tf.placeholder(tf.float32, [None, 784], name="X")  # input data가 들어올 자리
Y = tf.placeholder(tf.float32, [None, 10], name="Y")  # 정답이 들어올 자리, [0 0 0 0 0 0 0 0 0 1] one-hot encoding 형태
# *hint* dropout 확률을 위한 placeholder

W1 = tf.get_variable("W1", shape=[784, 300])  # 첫 번째 층의 weight matrix
b1 = tf.Variable(tf.random_normal([300]))  # 첫 번째 층의 bias
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Affine 연산 및 activation
# *hint* tf.nn.dropout으로 dropout 적용
# *hint* tf.nn.l2_loss로 parameter들의 l2_norm 값 누적

W2 = tf.get_variable("W2", shape=[300, 200])  # *hint* weight initialization을 위해 "initializer" 파라미터에 특정 초기화 기법을 입력
b2 = tf.Variable(tf.random_normal([200]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# *hint* tf.nn.dropout으로 dropout 적용
# *hint* tf.nn.l2_loss로 parameter들의 l2_norm 값 누적

W3 = tf.get_variable("W3", shape=[200, 10])
b3 = tf.Variable(tf.random_normal([10]))
# *hint* tf.nn.l2_loss로 parameter들의 l2_norm 값 누적

hypothesis = tf.nn.xw_plus_b(L2, W3, b3, name="hypothesis")  # L2W3 + b3
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# hypothesis를 softmax를 통해 확률값으로 변형, Y와 비교해 cross_entropy error 계산
# *hint* cost에 weight decay 적용

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,
                                                                  1))  # Y = [0,0,0,1,0,0,0,0,0,0], argmax로 실제 정답 인덱스 표현 e.g., argmax([0,0,0,1,0,0,0,0,0,0]) -> 3
# equal 함수는 정답 예측 여부를 판단, e.g., tf.equal(3,3) = True, tf.equal (8,3) =False
accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                  tf.float32))  # cast 함수는 형변환 함수, e.g., True -> 1.0 False -> 0.0, # 0.0 1.0 0.0 1.0 1.0 1.0...
# reduce_mean으로 batch 안에 있는 data들의 평균 정확도를 계산
summary_op = tf.summary.scalar("accuracy", accuracy)

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # back propagation

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 모든 parameter 값 초기화

training_epochs = 30
batch_size = 100

# 저장 directory와 tensor board 시각화를 위한 코드
# ========================================================================
timestamp = str(int(time.time()))  # runs/1578546654/checkpoints/
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# ========================================================================

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)  # 저장 객체 saver, max_to_keep은 저장하고자 하는 모델(시점) 개수

max = 0
early_stopped = 0
start_time = time.time()
for epoch in range(training_epochs):
    avg_cost = 0  # 전체 training data에 대한 평균 loss를 저장할 변수 초기화
    total_batch = int(mnist.train.num_examples / batch_size)  # iteration 55000/ 100 = 550

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(
            batch_size)  # batch size 단위로 input과 정답 리턴, e.g., (100, 784), (100, 10),
        feed_dict = {X: batch_xs, Y: batch_ys}  # placeholder에 실제 data를 먹여주기 위한 dictionary
        # *hint* dropout 확률을 placeholder에 추가
        c, _, a = sess.run([cost, optimizer, summary_op],
                           feed_dict=feed_dict)  # sess.run을 통해 원하는 operation 실행 및 결과값 return
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
    # ========================================================================
    train_summary_writer.add_summary(a, epoch)  ##
    val_accuracy, summaries = sess.run([accuracy, summary_op],
                                       feed_dict={X: mnist.validation.images, Y: mnist.validation.labels})
    # *hint* dropout 확률을 placeholder에 추가
    val_summary_writer.add_summary(summaries, epoch)  ##
    # ========================================================================

    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max:  # validation accuracy가 경신될 때
        max = val_accuracy
        early_stopped = epoch + 1  # early stopping 된 시점
        saver.save(sess, checkpoint_prefix,
                   global_step=early_stopped)  # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
training_time = (time.time() - start_time) / 60
print('Learning Finished!')
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)
print('training time: ', training_time)
