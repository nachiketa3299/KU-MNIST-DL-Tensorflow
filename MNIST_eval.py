import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


# 여러 개의 모델을 테스트하기 위해 플래그들을 루프마다 지워주기 위한 함수(인터넷에서 복붙한 것)
def del_all_flags(_FLAGS):
    flags_dict = _FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        _FLAGS.__delattr__(keys)

# mnist 데이터 가져오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

# Train된 모델이 저장된 파일 경로 찾아서 가져오기
filenames = os.listdir(os.path.join(os.path.curdir, 'runs'))

# 테스트 셋에 대한 결과를 저장할 eval_log.txt 열기
file = open('./INFOS/eval_log.txt', 'w')
for filename in filenames:
    # 플래그 지워 주고
    del_all_flags(tf.flags.FLAGS)
    # 루프마다 그래프 초기화 해주고
    tf.reset_default_graph()
    # 경로에서 체크포인트 불러옴
    tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{filename}/checkpoints", "Checkpoint directory from training run")
    FLAGS = tf.flags.FLAGS
    # 해당 모델에서 가장 validation accuracy가 높은 시점 load
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph(f"{checkpoint_file}.meta")
            # 저장했던 모델 load
            saver.restore(sess, checkpoint_file)

            # Operation 가져오기
            X = graph.get_operation_by_name("X").outputs[0]
            Y = graph.get_operation_by_name("Y").outputs[0]
            # Dropout을 위한 플레이스홀더 선언
            dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

            hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]
            correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # 테스트 데이터를 먹일 딕셔너리 선언
            feed_dict = {X:mnist.test.images, Y:mnist.test.labels, dropout_prob: 0.0}
            # Feed!
            test_accuracy = sess.run(accuracy, feed_dict=feed_dict)

            # eval_log.txt에 기록
            file.write(f'{filename}\t{test_accuracy}\n')
            print('- Timestamp        : ', filename)
            print('- Test Max Accuracy:', test_accuracy)
    sess.close()
file.close()



