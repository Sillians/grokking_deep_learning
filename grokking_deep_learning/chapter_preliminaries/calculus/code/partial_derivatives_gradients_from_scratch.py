x = 2.0
y_true = 7.0
w = 1.0
b = 0.0
learning_rate = 0.01
epochs = 10

# training loop
for epoch in range(epochs):
    y_pred = w * x + b
    loss = 0.5 * (y_pred - y_true) ** 2

    dL_dy_pred = y_pred - y_true
    dL_dw = dL_dy_pred * x
    dL_db = dL_dy_pred

    w = w - learning_rate * dL_dw
    b = b - learning_rate * dL_db

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")