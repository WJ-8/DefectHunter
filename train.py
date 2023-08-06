import keras_metrics as km
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, concatenate, \
    Dropout, Reshape
from tensorflow.keras.optimizers import Adam

from network.attention import SinusoidalPositionEmbedding
from network.network import ConformerEncoder

y_train = np.load("data/dataset/train_y.npy")
x_train_emb = np.load("data/dataset/train_emb.npy")
x_train_ast = np.load("data/dataset/train_ast.npy")
x_train_dfg = np.load("data/dataset/train_dfg.npy")
x_train_cfg = np.load("data/dataset/train_cfg.npy")

# 加载验证集
y_val = np.load("data/dataset/valid_y.npy")
x_val_emb = np.load("data/dataset/valid_emb.npy")
x_val_ast = np.load("data/dataset/valid_ast.npy")
x_val_dfg = np.load("data/dataset/valid_dfg.npy")
x_val_cfg = np.load("data/dataset/valid_cfg.npy")

# 加载测试集

y_test = np.load("data/dataset/test_y.npy")
x_test_emb = np.load("data/dataset/test_emb.npy")
x_test_ast = np.load("data/dataset/test_ast.npy")
x_test_dfg = np.load("data/dataset/test_dfg.npy")
x_test_cfg = np.load("data/dataset/test_cfg.npy")
# 模型输入
input_emb = Input(shape=(1, 768))
input_dfg = Input(shape=(200, 200))
input_cfg = Input(shape=(200, 200))
input_ast = Input(shape=(200, 200))
# 正余弦编码
sin_emb = SinusoidalPositionEmbedding(output_dim=768, merge_mode="mul")(input_emb)
sin_dfg = SinusoidalPositionEmbedding(output_dim=200, merge_mode="mul")(input_dfg)
sin_cfg = SinusoidalPositionEmbedding(output_dim=200, merge_mode="mul")(input_cfg)
sin_ast = SinusoidalPositionEmbedding(output_dim=200, merge_mode="mul")(input_ast)
shape_emb1 = Reshape((1, 768))(sin_emb)

conforencode_emb = ConformerEncoder(embedding_dim=768, num_heads=8, feed_forward_dim=1024, num_blocks=12)(shape_emb1)
conforencode_ast = ConformerEncoder(embedding_dim=200, num_heads=8, feed_forward_dim=1024, num_blocks=12)(sin_ast)
conforencode_cfg = ConformerEncoder(embedding_dim=200, num_heads=8, feed_forward_dim=1024, num_blocks=12)(sin_cfg)
conforencode_dfg = ConformerEncoder(embedding_dim=200, num_heads=8, feed_forward_dim=1024, num_blocks=12)(sin_dfg)

shape_emb2 = Reshape((1, 768))(conforencode_emb)

flatten_emb = Flatten()(shape_emb2)
flatten_ast = Flatten()(conforencode_ast)
flatten_dfg = Flatten()(conforencode_dfg)
flatten_cfg = Flatten()(conforencode_cfg)

merged = concatenate([flatten_emb, flatten_dfg, flatten_cfg, flatten_ast])

dense_1 = Dense(1024, activation="relu")(merged)
drop_1 = Dropout(0.1)(dense_1)
z = Dense(2, activation="softmax")(drop_1)

model = Model(inputs=[input_emb, input_dfg, input_cfg, input_ast], outputs=z)
model.compile(optimizer=Adam(learning_rate=1e-5), loss=["sparse_categorical_crossentropy"],  # 8e-7
              metrics=["accuracy", km.sparse_categorical_f1_score(), km.sparse_categorical_precision(),
                       km.sparse_categorical_recall()])
model.summary()
checkpoints = ModelCheckpoint(filepath='model/ck/weights.{epoch:04d}.hdf5', monitor="val_loss", verbose=1,
                              save_weights_only=True, period=1)

history = model.fit(
    [x_train_emb, x_train_dfg, x_train_cfg, x_train_ast], y_train,
    validation_data=([x_val_emb, x_val_dfg, x_val_cfg, x_val_ast], y_val),
    epochs=50, batch_size=64, verbose=2, callbacks=[checkpoints])
score = model.evaluate([x_test_emb, x_test_dfg, x_test_cfg, x_test_ast], y_test, verbose=0, batch_size=128)
print(score)
print(model.metrics_names)
