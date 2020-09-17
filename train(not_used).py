import time
import random
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

### HYPERPARAMETERS STARTS
SEED = 1019
INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10
HIDDEN_L_SIZE = [300, 200]
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.5
LAMBDA = 0.0001
TRAINING_EPOCHS = 30
BATCH_SIZE = 100

### HYPERPARAMETERS ENDS

tf.set_random_seed(SEED)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)
# one_hot = True, 4-> 0, 0, 0, 0, 1, 0, 0, 0, 0, 0

X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="X")  # input data가 들어올 자리
Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="Y")  # 정답이 들어올 자리, [0 0 0 0 0 0 0 0 0 1] one-hot encoding 형태
# TODO [complete] *hint* dropout 확률을 위한 placeholder
dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
# dropout = tf.placeholder()

# 1st Hidden Layer
W1 = tf.get_variable("W1", shape=[INPUT_SIZE, HIDDEN_L_SIZE[0]], initializer=tf.contrib.layers.xavier_initializer())  # 첫 번째 층의 weight matrix
b1 = tf.Variable(tf.random_normal([HIDDEN_L_SIZE[0]]), name="b1")  # 첫 번째 층의 bias
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Affine 연산 및 activation
# TODO [complete] *hint* tf.nn.dropout으로 dropout 적용
L1 = tf.nn.dropout(L1, rate=dropout_prob)
# TODO *hint* tf.nn.l2_loss로 parameter들의 l2_norm 값 누적
weight_decay = tf.nn.l2_loss(W1)

# 2nd Hidden Layer
W2 = tf.get_variable("W2", shape=[HIDDEN_L_SIZE[0], HIDDEN_L_SIZE[1]], initializer=tf.contrib.layers.xavier_initializer())
# TODO [complete] *hint* weight initialization을 위해 "initializer" 파라미터에 특정 초기화 기법을 입력
b2 = tf.Variable(tf.random_normal([HIDDEN_L_SIZE[1]]), name="b2")
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# TODO [complete] *hint* tf.nn.dropout으로 dropout 적용
L2 = tf.nn.dropout(L2, rate=dropout_prob)
# TODO *hint* tf.nn.l2_loss로 parameter들의 l2_norm 값 누적
weight_decay = tf.add(tf.nn.l2_loss(W2), weight_decay)

# 3rd Hidden Layer
W3 = tf.get_variable("W3", shape=[HIDDEN_L_SIZE[1], OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([OUTPUT_SIZE]), name="b3")
# TODO *hint* tf.nn.l2_loss로 parameter들의 l2_norm 값 누적
weight_decay = tf.add(tf.nn.l2_loss(W3), weight_decay)

hypothesis = tf.nn.xw_plus_b(L2, W3, b3, name="hypothesis")  # L2W3 + b3
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))와 같은 의미
# hypothesis를 softmax를 통해 확률값으로 변형, Y와 비교해 cross_entropy error 계산
# reduced_mean은 모든 차원을 제거하고 단 하나의 스칼라 평균을 리턴함
# corss_entropy는 인간이 정한 답(1)에 대해 모델이 잘못된 판단을 내린 경우 페널티를 주는 로스함수이다. 답이 아닌 부분(0)에 대해서는 관심이 없다.
# 여기까지 에러의 평균(cost)이 계산됨.

# TODO *hint* cost에 weight decay 적용
cost += LAMBDA * weight_decay

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Y = [0,0,0,1,0,0,0,0,0,0], argmax로 실제 정답 인덱스 표현 e.g., argmax([0,0,0,1,0,0,0,0,0,0]) -> 3
# tf.argmax(hypothesis)의 리턴값은 모델이 예측한 가장 정답에 가까운 인덱스의 원 핫 인코딩.
# tf.argmax(Y, 1)의 리턴값은 정답 인덱스의 원 핫 인코딩.
# equal 함수는 정답 예측 여부를 판단, e.g., tf.equal(3,3) = True, tf.equal (8,3) =False
# correct_prediction은 그래서 모델이 제대로 맞추었는지, 맞추지 못했는지를 기록하게 된다.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# cast 함수는 형변환 함수, e.g., True -> 1.0 False -> 0.0, # 0.0 1.0 0.0 1.0 1.0 1.0...
# reduce_mean으로 batch 안에 있는 data들의 평균 정확도를 계산

summary_op = tf.summary.scalar("accuracy", accuracy)

# back propagation
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 모든 parameter 값 초기화

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
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
# 저장 객체 saver, max_to_keep은 저장하고자 하는 모델(시점) 개수

max_accuracy = 0
early_stopped = 0

start_time = time.time()
for epoch in range(TRAINING_EPOCHS):
    # 전체 training data에 대한 평균 loss를 저장할 변수 초기화
    avg_cost = 0
    # iteration 55000/ 100 = 550
    total_batch = int(mnist.train.num_examples / BATCH_SIZE)

    a = None
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # batch size 단위로 input과 정답 리턴, e.g., (100, INPUT_SIZE), (100, 10),
        # TODO [complete] *hint* dropout 확률을 placeholder에 추가
        feed_dict = {X: batch_xs, Y: batch_ys, dropout_prob: DROPOUT_PROB}  # placeholder에 실제 data를 먹여주기 위한 dictionary
        c, _, a = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
        # sess.run을 통해 원하는 operation 실행 및 결과값 return
        avg_cost += c / total_batch


    # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
    # ========================================================================
    train_summary_writer.add_summary(a, epoch)  ##
    # TODO [complete] *hint* dropout 확률을 placeholder에 추가
    val_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, dropout_prob: DROPOUT_PROB})
    val_summary_writer.add_summary(summaries, epoch)  ##
    # ========================================================================

    print(f'> Epoch: {format(epoch + 1, "04")}/{format(TRAINING_EPOCHS, "04")}\ttraining_cost={"{:.9f}".format(avg_cost)}\tvalidation_accuracy={val_accuracy}')

    if val_accuracy > max_accuracy:  # validation accuracy가 경신될 때
        max_accuracy = val_accuracy
        early_stopped = epoch + 1  # early stopping 된 시점
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)
        # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping

training_time = (time.time() - start_time) / 60

print(f'\n** Learning Finished! (SEED={SEED})')
print('- Training time:          ', training_time)
print('- Early stopped epoch:    ', early_stopped)
print('- Validation Max Accuracy:', max_accuracy)
print('- Timestamp:              ', timestamp)

