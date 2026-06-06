import numpy as np # Mathematical calculations ke liye NumPy library
import matplotlib.pyplot as plt # Graph aur charts banane ke liye Matplotlib library

# 1. Setup the training data (Humara asli data)
# x_train: Ghar ka size (1000 sqft mein)
# y_train: Ghar ki actual keemat (1000s dollars mein)
x_train = np.array([1.0, 2.0]) # Pehla ghar 1000 sqft, doosra 2000 sqft
y_train = np.array([300.0, 500.0]) # Pehle ki keemat $300k, doosre ki $500k

# 2. Define the model function (Prediction nikalne ka math)
def compute_model_output(x, w, b):
    """
    Yeh function linear model (y = wx + b) ki calculation karta hai.
    """
    m = x.shape[0] # Pata lagata hai ki total kitne data points hain (yahan m=2 hai)
    f_wb = np.zeros(m) # Ek khali array banata hai: [0., 0.] predictions store karne ke liye
    
    for i in range(m): # Har ek ghar ke data par bari-bari jayega
        f_wb[i] = w * x[i] + b # Prediction calculate karke array mein daal dega
        
    return f_wb # Poori prediction list wapas bhej dega

# 3. Set parameters (Model ke weight aur bias)
w = 300 # Weight: Line ki slope/dhalan (Har 1 unit x badhne par y kitna badhega)
b = 0   # Bias: Y-intercept (Line y-axis par kahan se start hogi)

# 4. Generate predictions
tmp_f_wb = compute_model_output(x_train, w, b) # Function chala kar apni predictions nikali

# 5. Plot the results (Graph banana)
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction') # Apni predictions ko blue line ('b') se draw kiya
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values') # Asli prices ko red ('r') 'x' marks se draw kiya

# Graph ki Formatting (Labels aur title lagana)
plt.title("Housing Prices") # Graph ke upar ka main title
plt.ylabel('Price (in 1000s of dollars)') # Y-axis (vertical) ka label
plt.xlabel('Size (1000 sqft)') # X-axis (horizontal) ka label
plt.legend() # Ek chhota sa dabba dikhayega jo batayega blue line aur red x kya hain
plt.show() # Final graph ko screen par display karega

# 6. Predict a new value (Ek naye ghar ki keemat nikalna)
x_new = 1.2 # Ek naya ghar jo 1200 sqft ka hai (units ke hisaab se 1.2)
prediction = w * x_new + b # Naye ghar par y = wx + b ka formula lagaya
print(f"Prediction for a house with 1200 sqft: ${prediction:.0f} thousand dollars") # Result ko screen par print kiya
