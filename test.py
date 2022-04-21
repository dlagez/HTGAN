from transformer import SwinTransformer
import numpy as np
# tr = SwinTransformer()


list = [i for i in range(2457600)]
a = np.array(list).reshape(200, 3, 64, 64)
b = a.reshape(200, 64, 64, 3)