import pprint

import jax
import jax.numpy as jnp
from jaxopt.implicit_diff import custom_root

# Define the function whose root we want to find, e.g., f(x) = x^2 - 2
def my_function(x):
    return x**2 - 2

# Define an optimality function that only depends on x
def optimality_fn(x, diff):
    return my_function(x) - diff

# Define a simple solver that takes only the initial guess as an argument
def newton_solver(initial_guess, tol, diff):
    x = initial_guess
    max_iter = 100
    for _ in range(max_iter):
        fx = my_function(x) - diff
        dfx = 2 * x  # Derivative of f(x) = x^2 - 2 is 2x
        x -= fx / dfx  # Newton update
        if abs(fx) < tol:
            return x
    return x

# Decorate the solver with custom_root to make it differentiable
decorated_solver = custom_root(optimality_fn, has_aux=False)(newton_solver)

# # Use the decorated solver to find the root
initial_guess = 3.0
tol = 1e-5
diff = 1.0
root = decorated_solver(initial_guess, tol, diff)
print("Root:", root)

# Wrap decorated_solver in a lambda that only takes `diff` as input
root_with_respect_to_diff = lambda diff1: decorated_solver(initial_guess, tol, diff1)

# Calculate the gradient of the root with respect to `diff`
root_gradient_fn = jax.grad(root_with_respect_to_diff, argnums=0)
root_gradient = root_gradient_fn(diff)
print("Gradient of the root with respect to `diff`:", root_gradient)