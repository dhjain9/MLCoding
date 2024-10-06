import numpy as np

# here a => a + kb
#      b => b
# Thus T is
#      [1 k]
#      [0 1]
def T_shear(v, k):
    A = np.array([[1, k], [0,1]])
    w = A @ v
    
    return w
    
    
v = np.array([[1], [1]])

w = T_shear(v, 3)

print("For Vector:\n v= \n", v, 
      "\n\n Result of the shear transformation with k=3 (matrix form):\n",
      w)
