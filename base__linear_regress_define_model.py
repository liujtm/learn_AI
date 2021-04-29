import tensorflow as tf

tf.keras.layers.Conv2D
tf.nn.embedding_lookup
tf.nn.nce_loss
# batch_size=2, data_dim=3, shape为(2,3)
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2个样本，每个样本shape(3)
y = tf.constant([[10.0], [20.0]])

# 1. 继承tf.keras.Model 2. __init__()里定义各个层layer 3. call()里定义输入输出
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense为全连接层, y = activation_func(X * kernel + bias)
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(), # 根据输入X (shape (2,3))自动推断kernel shape，这里为 (3,1)
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
tf.saved_model.save(model, "saved/")
tf.saved_model.load
