# Tiny neural network demonstrating:
# forward pass → loss → backpropagation → gradient → weight update

# Our training data
x = 3.0
target = 6.0

# Start with a bad weight
w = 0.5

# Learning rate
learning_rate = 0.1

for step in range(20):

    # -------------------------
    # 1. FORWARD PASS
    # -------------------------

    prediction = w * x

    # -------------------------
    # 2. CALCULATE LOSS
    # -------------------------

    error = prediction - target
    loss = error ** 2

    # -------------------------
    # 3. BACKPROPAGATION
    # -------------------------
    #
    # loss = (prediction - target)^2
    # prediction = w * x
    #
    # Using the chain rule:
    #
    # d(loss)/d(w)
    # = 2 * error * x

    gradient = 2 * error * x

    # -------------------------
    # 4. UPDATE WEIGHT
    # -------------------------

    w = w - learning_rate * gradient

    print(
        f"Step {step:2d} | "
        f"Weight: {w:.4f} | "
        f"Prediction: {prediction:.4f} | "
        f"Loss: {loss:.4f} | "
        f"Gradient: {gradient:.4f}"
    )