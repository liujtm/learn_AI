import tensorflow as tf
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)  # 5个样本，每个样本是一个标量
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b  # 会自动把标量a,b的维度补齐到和X一致, X和y_pred为向量（1维数组）（长度5）
        loss = tf.reduce_sum(tf.square(y_pred - y))  # reduce_sum 对 数组 内所有元素求和，loss为标量（0维数组），对5个样本loss之和整体优化  Full batch
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)  # grads为向量（长度2），即[grad_a, grad_b]，参数a、b分别的梯度
    # TensorFlow自动根据梯度更新参数
    # a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))  #zip()把(grad_a, a), (grad_b, b)分别打包
    print("%s: [a, b]=%s" % (str(e), str(variables)))
