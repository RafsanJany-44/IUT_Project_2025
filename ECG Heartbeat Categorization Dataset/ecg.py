import numpy as np
import matplotlib.pyplot as plt

# Load the prepared dataset (374 columns: 187 X + 187 Y)
xy = np.loadtxt(r"C:\Users\user\Documents\GitHub\IUT_Project_2025\ECG Heartbeat Categorization Dataset\ecg_abnormal_denoise.csv", delimiter=",")

print("xy shape:", xy.shape)   # (4046, 374)


n_features = 187

X = xy[:, :n_features]       # original ECG
Y = xy[:, n_features:]       # smoothed ECG

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,      # 20% for test
    random_state=42,    # reproducible
    shuffle=True        # always shuffle ECG beats
)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes:",  X_test.shape, Y_test.shape)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




#model_3
l0    = tf.keras.layers.Dense(units=32,  input_shape=[n_features])
l1    = tf.keras.layers.Dense(units=64)
l_out = tf.keras.layers.Dense(units=n_features)   # keep this linear

model_eff = tf.keras.Sequential([l0, l1,  l_out])



model = model_eff


model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001)) # type: ignore
# 4. Train
history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,   # 10% of training used for validation
    verbose=0 # type: ignore
)

test_loss = model.evaluate(X_test, Y_test, verbose=0)
print("Test MSE:", test_loss)

Y_pred = model.predict(X_test)



idx = 0
orig = X_test[idx]
true = Y_test[idx]
pred = Y_pred[idx]

plt.plot(orig, label="Original ECG")
plt.plot(true, label="Target Smoothed ECG")
plt.plot(pred, '--', label="Model Output")
plt.legend()
plt.show()
