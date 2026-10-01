# Neural Network Learning Journey
# --------------------------------
# Version 2:
# forward pass → loss → backpropagation → gradient → weight update
#
# NEW:
# - Bias (b)
# - Gradient for the weight
# - Gradient for the bias

x = 3.0
target = 6.0

w = 0.5
b = 0.0

learning_rate = 0.01

for step in range(50):

    # -----------------
    # 1. Forward pass
    # -----------------
    prediction = w * x + b

    # -----------------
    # 2. Calculate error
    # -----------------
    error = prediction - target

    # -----------------
    # 3. Calculate loss
    # -----------------
    loss = error ** 2

    # -----------------
    # 4. Backpropagation
    # -----------------

    # d(loss) / d(weight)
    gradient_w = 2 * error * x

    # d(loss) / d(bias)
    gradient_b = 2 * error

    # -----------------
    # 5. Update parameters
    # -----------------
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    print(
        f"Step {step:2d} | "
        f"Weight: {w:.4f} | "
        f"Bias: {b:.4f} | "
        f"Prediction: {prediction:.4f} | "
        f"Loss: {loss:.4f} | "
        f"Grad W: {gradient_w:.4f} | "
        f"Grad B: {gradient_b:.4f}"
    )
