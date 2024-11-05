import jax


# Define a function of two variables
def g(x, y):
    return x**2 + y**2

# Get the gradient of g with respect to the first argument (x)
dg_dx = jax.grad(g, argnums=0)

# Get the gradient of g with respect to the second argument (y)
dg_dy = jax.grad(g, argnums=1)

# Calculate gradients at a specific point (x=3.0, y=4.0)
x_value, y_value = 3.0, 4.0
gradient_x = dg_dx(x_value, y_value)
gradient_y = dg_dy(x_value, y_value)

print("g(x, y) at (x=3.0, y=4.0):", g(x_value, y_value))
print("dg/dx at (x=3.0, y=4.0):", gradient_x)
print("dg/dy at (x=3.0, y=4.0):", gradient_y)
