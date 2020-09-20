import time
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import hyperparameters as hp

# mnist 데이터 받아오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

# 하이퍼파라미터 선언(프리셋 적용)

presets = list(range(1, 38))
# 프리셋을 바꿔가며 자동으로 train
for p in presets:
    # 루프마다 이전 그래프들을 초기화시킴
    tf.reset_default_graph()
    h_p = hp.hyperparameters(preset=p)
    # SEED만 같다면 같은 결과값이 나오도록 설정
    tf.set_random_seed(h_p.SEED)

    ## 레이어 선언
    X = tf.placeholder(tf.float32, [None, h_p.INPUT_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, h_p.OUTPUT_SIZE], name="Y")
    # 모든 W, B, L을 담아둘 리스트 선언
    W = []
    B = []
    L = []
    act_func = None
    weight_decay = None
    cost = None
    hypothesis = None
    optimizer = None
    dropout_prob = None

    # Activation Function 결정
    if h_p.ACT_FUNC == 'relu':
        act_func = tf.nn.relu

    for i in range(0, h_p.N_OF_HIDDEN_L):
        # W, B, L의 이름 결정
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

        # Bias 선언 및 초기화 (initialize를 무작위로 진행)
        if i != h_p.N_OF_HIDDEN_L - 1:
            B.append(tf.Variable(tf.random_normal([h_p.HIDDEN_L_SIZE[i]]), name=b_name))
        else:
            B.append(tf.Variable(tf.random_normal([h_p.OUTPUT_SIZE]), name=b_name))

        # Graph 추가 후 활성화함수 적용.
        if i == 0:
            L.append(act_func(tf.matmul(X, W[i]) + B[i]))
            L[i] = tf.nn.dropout(L[i], rate=h_p.DROPOUT)
        elif i != h_p.N_OF_HIDDEN_L - 1:
            L.append(act_func(tf.matmul(L[i - 1], W[i]) + B[i]))
            L[i] = tf.nn.dropout(L[i], rate=h_p.DROPOUT)
        # Output으로 나가는 마지막 레이어에서 cost를 계산
        else:
            hypothesis = tf.nn.xw_plus_b(L[i - 1], W[i], B[i], name='hypothesis')
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

    # Dropout을 위한 placeholder 선언
    dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

    # cost에 Weight Decay 적용
    for i in range(len(W)):
        if i == 0:
            weight_decay = tf.nn.l2_loss(W[i])
        else:
            weight_decay = tf.add(tf.nn.l2_loss(W[i]), weight_decay)
        # WeightDecay가 적용된 cost. WEIGHT_DECAY가 0이면 적용되지 않은 것과 같다.
        cost += h_p.WEIGHT_DECAY * weight_decay

    # Train데이터와 정답과 가장 가까운 index끼리 비교하여 같은지 같지 않은지 리턴.
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_op = tf.summary.scalar("accuracy", accuracy)

    # 사용할 Optimizer 설정 및 Backpropagation
    if h_p.OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=h_p.LEARNING_RATE).minimize(cost)
    elif h_p.OPTIMIZER == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=h_p.LEARNING_RATE).minimize(cost)


    # Session Initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ## 저장 directory와 tensor board 시각화를 위한 코드
    timestamp = str(int(time.time()))  # runs/{timestamp}/checkpoints/

    # ./runs/{timestampe}
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

    # ./runs/{timestamp}/summaries/train
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # ./runs/{timestamp}/summaries/dev
    val_summary_dir = os.path.join(out_dir, "summaries", "dev")
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

    # ./runs/{timestamp}/checkpoints
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    # ./runs/{timestamp}/checkpoints/model
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    # 저장 객체 saver, max_to_keep은 저장하고자 하는 모델(시점) 개수

    max_accuracy = 0
    early_stopped = 0

    # Training log를 기록하기 위한 txt 파일 생성 ./log.txt
    filename = f"./INFOS/training_log.txt"
    file = open(filename, 'a')
    file.write(f"{timestamp}\n")

    start_time = time.time()
    for epoch in range(h_p.TRAINING_EPOCH):
        # 전체(모든 Epoch 통합) training data에 대한 평균 loss를 저장할 변수 초기화
        avg_cost, avg_acc = 0, 0
        # 총 iteration 횟수 = Mnist 트레이닝 데이터 숫자 / 배치사이즈
        total_batch = int(mnist.train.num_examples / h_p.BATCH_SIZE)

        # Iteration
        for i in range(total_batch):
            # batch_xs는 배치 인풋, batch_ys는 배치 정답 (크기는 모두 batch size)
            batch_xs, batch_ys = mnist.train.next_batch(h_p.BATCH_SIZE)


            # Training Model에 먹일 데이터 선언. DROPOUT이 적용되지 않을 경우 확률이 0이 됨.
            feed_dict = {X: batch_xs, Y: batch_ys, dropout_prob: h_p.DROPOUT}

            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
            # optimizer = tf.train.AdamOptimizer(learning_rate=h_p.LEARNING_RATE).minimize(cost)
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            _cost, _, _accuracy = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
            # cost 에서 Loss들이 구해지고, 그것을 바탕으로 optimizer에서 Backpropagation이 이루어짐, 그리고 모델이 얼마나 정확한 출력을 리턴했는지 accuracy를 측정
            # sess.run을 통해 원하는 operation 실행 및 결과값 return
            avg_cost += _cost / total_batch
            avg_acc += _accuracy / total_batch


        # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=avg_acc)])
        train_summary_writer.add_summary(train_summary, epoch)  ##

        # summary_op = tf.summary.scalar("accuracy", accuracy)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Validation을 위한 feed_dict. Dropout 적용도 동일하다.
        feed_dict_val = {X: mnist.validation.images,Y: mnist.validation.labels, dropout_prob: 0.0}
        val_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict=feed_dict_val)
        val_summary_writer.add_summary(summaries, epoch)

        # for Terminal output
        print(f'> Preset: {format(p, "02")}  Epoch: {format(epoch + 1, "04")}/{format(h_p.TRAINING_EPOCH, "04")}  training_cost={"{:.9f}".format(avg_cost)}\tvalidation_accuracy={val_accuracy}')
        # ./train_log.txt output
        file.write(f'preset({h_p.PRESET})\t{format(epoch + 1, "04")}/{format(h_p.TRAINING_EPOCH, "04")}  {"{:.9f}".format(avg_cost)}  {val_accuracy}\n')

        if val_accuracy > max_accuracy:  # validation accuracy가 경신될 때
            max_accuracy = val_accuracy
            early_stopped = epoch + 1  # early stopping 된 시점
            saver.save(sess, checkpoint_prefix, global_step=early_stopped)
            # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping

    training_time = (time.time() - start_time) / 60

    # for Terminal output
    print(f'\n** Training Finished! (SEED={h_p.SEED})')
    h_p.pprint()
    print(">> Training Results")
    print('- Training time          :', training_time)
    print('- Early stopped epoch    :', early_stopped)
    print('- Validation Max Accuracy:', max_accuracy)
    print('- Timestamp              :', timestamp)

    file.close()
    filename = "./INFOS/overral_info.txt"
    file = open(filename, 'a')
    if os.path.getsize(filename) == 0:
        file.write("preset\tbatch_size\tactivation_function\t#_of_layers\tlayer_size\ttraining_epoch\tweight_init\toptimizer\tweight_decay\tdropout\ttraining_time\tearly_stopping\tval_maxAcc\ttimestamp\n")
    file.write(f"{h_p.PRESET}\t{h_p.BATCH_SIZE}\t{h_p.ACT_FUNC}\t{h_p.N_OF_HIDDEN_L}\t{h_p.HIDDEN_L_SIZE}\t{h_p.TRAINING_EPOCH}\t{h_p.WEIGHT_INIT}\t{h_p.OPTIMIZER}\t{h_p.LEARNING_RATE}\t{h_p.WEIGHT_DECAY}\t{h_p.DROPOUT}\t{training_time}\t{early_stopped}\t{max_accuracy}\t{timestamp}\n")
    file.close()
    sess.close()

