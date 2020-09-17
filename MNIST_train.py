import time
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import hyperparameters as hp

# mnist 데이터 받아오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

# 하이퍼파라미터 선언(프리셋 적용)
presets = [1, 2, 3, 4, 5, 6, 7, 8]
for p in presets:
    tf.reset_default_graph()
    h_p = hp.hyperparameters(preset=p)
    tf.set_random_seed(h_p.SEED)



    ## 레이어 선언
    X = tf.placeholder(tf.float32, [None, h_p.INPUT_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, h_p.OUTPUT_SIZE], name="Y")
    W = []
    B = []
    L = []
    act_func = None
    weight_decay = None
    cost = None
    hypothesis = None
    optimizer = None
    dropout_prob = None

    if h_p.ACT_FUNC == 'relu':
        act_func = tf.nn.relu

    for i in range(0, h_p.N_OF_HIDDEN_L):
        # Weight Matrix의 이름과 Bias의 이름 결정
        w_name = "W" + str(i + 1)
        b_name = "B" + str(i + 1)
        l_name = "L" + str(i + 1)
        # Weight Matrix의 Layer별 Shape 결정
        if i == 0:
            w_shape = [h_p.INPUT_SIZE, h_p.HIDDEN_L_SIZE[i]]
        elif i != h_p.N_OF_HIDDEN_L - 1:
            w_shape = [h_p.HIDDEN_L_SIZE[i - 1], h_p.HIDDEN_L_SIZE[i]]
        else:
            w_shape = [h_p.HIDDEN_L_SIZE[i - 1], h_p.OUTPUT_SIZE]
        # Weight Initialization의 방법 결정
        if h_p.WEIGHT_INIT == "he":
            w_init = tf.contrib.layers.variance_scaling_initializer()
        elif h_p.WEIGHT_INIT == "xavier":
            w_init = tf.contrib.layers.xavier_initializer()
        else:
            w_init = None

        w_temp = tf.get_variable(w_name, shape=w_shape, initializer=w_init)
        W.append(w_temp)

        # Bias 선언
        if i != h_p.N_OF_HIDDEN_L - 1:
            B.append(tf.Variable(tf.random_normal([h_p.HIDDEN_L_SIZE[i]]), name=b_name))
        else:
            B.append(tf.Variable(tf.random_normal([h_p.OUTPUT_SIZE]), name=b_name))

        # Graph 추가 후 활성화함수 적용.
        if i == 0:
            L.append(act_func(tf.matmul(X, W[i]) + B[i]))
            if h_p.DROPOUT is not None:
                L[i] = tf.nn.dropout(L[i], rate=h_p.DROPOUT)
        elif i != h_p.N_OF_HIDDEN_L - 1:
            L.append(act_func(tf.matmul(L[i - 1], W[i]) + B[i]))
            if h_p.DROPOUT is not None:
                L[i] = tf.nn.dropout(L[i], rate=h_p.DROPOUT)
        # Output으로 나가는 마지막 레이어에서 cost를 계산
        else:
            hypothesis = tf.nn.xw_plus_b(L[i - 1], W[i], B[i], name='hypothesis')
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

    if h_p.DROPOUT is not None:
        dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

    if h_p.WEIGHT_DECAY is not None:
        for i in range(len(W)):
            if i == 0:
                weight_decay = tf.nn.l2_loss(W[i])
            else:
                weight_decay = tf.add(tf.nn.l2_loss(W[i]), weight_decay)
        cost += h_p.WEIGHT_DECAY * weight_decay

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_op = tf.summary.scalar("accuracy", accuracy)

    # back propagation with optimizer
    if h_p.OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=h_p.LEARNING_RATE).minimize(cost)
    elif h_p.OPTIMIZER == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=h_p.LEARNING_RATE).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 저장 directory와 tensor board 시각화를 위한 코드
    # ========================================================================
    timestamp = str(int(time.time()))  # runs/{timestamp}/checkpoints/
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
    filename = f"./log.txt"
    file = open(filename, 'a')
    file.write(f"{timestamp}\n")

    for epoch in range(h_p.TRAINING_EPOCH):
        # 전체 training data에 대한 평균 loss를 저장할 변수 초기화
        avg_cost = 0
        avg_acc = 0
        # iteration 55000/ 100 = 550
        total_batch = int(mnist.train.num_examples / h_p.BATCH_SIZE)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(h_p.BATCH_SIZE)
            # batch size 단위로 input과 정답 리턴, e.g., (100, INPUT_SIZE), (100, 10),
            # TODO [complete] *hint* dropout 확률을 placeholder에 추가
            if h_p.DROPOUT is not None:
                feed_dict = {X: batch_xs, Y: batch_ys, dropout_prob: h_p.DROPOUT}
            else:
                feed_dict = {X: batch_xs, Y: batch_ys}
            c, _, a = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
            # sess.run을 통해 원하는 operation 실행 및 결과값 return
            avg_cost += c / total_batch
            avg_acc += a / total_batch


        # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
        # ========================================================================
        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=avg_acc)])
        train_summary_writer.add_summary(train_summary, epoch)  ##
        # TODO [complete] *hint* dropout 확률을 placeholder에 추가
        if h_p.DROPOUT is not None:
            val_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, dropout_prob: h_p.DROPOUT})
        else:
            val_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels})
        val_summary_writer.add_summary(summaries, epoch)  ##
        # ========================================================================

        print(f'> Preset: {p}/{len(presets)}  Epoch: {format(epoch + 1, "04")}/{format(h_p.TRAINING_EPOCH, "04")}  training_cost={"{:.9f}".format(avg_cost)}\tvalidation_accuracy={val_accuracy}')
        file.write(f'{format(epoch + 1, "04")}/{format(h_p.TRAINING_EPOCH, "04")}  {"{:.9f}".format(avg_cost)}  {val_accuracy}\n')

        if val_accuracy > max_accuracy:  # validation accuracy가 경신될 때
            max_accuracy = val_accuracy
            early_stopped = epoch + 1  # early stopping 된 시점
            saver.save(sess, checkpoint_prefix, global_step=early_stopped)
            # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping

    training_time = (time.time() - start_time) / 60


    print(f'\n** Training Finished! (SEED={h_p.SEED})')
    h_p.pprint()
    print(">> Training Results")
    print('- Training time          :', training_time)
    print('- Early stopped epoch    :', early_stopped)
    print('- Validation Max Accuracy:', max_accuracy)
    print('- Timestamp              :', timestamp)

    file.close()
    filename = "./overral_info.txt"
    file = open(filename, 'a')
    if os.path.getsize(filename) == 0:
        file.write("preset\tbatch_size\tactivation_function\t#_of_layers\tlayer_size\ttraining_epoch\tweight_init\toptimizer\tweight_decay\tdropout\ttraining_time\tearly_stopping\tval_maxAcc\ttimestamp\n")
    file.write(f"{h_p.PRESET}\t{h_p.BATCH_SIZE}\t{h_p.ACT_FUNC}\t{h_p.N_OF_HIDDEN_L}\t{h_p.HIDDEN_L_SIZE}\t{h_p.TRAINING_EPOCH}\t{h_p.WEIGHT_INIT}\t{h_p.OPTIMIZER}\t{h_p.WEIGHT_DECAY}\t{h_p.DROPOUT}\t{training_time}\t{early_stopped}\t{max_accuracy}\t{timestamp}\n")
    file.close()
    sess.close()

