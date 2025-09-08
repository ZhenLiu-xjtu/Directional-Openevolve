import numpy as np

#EVOLVE-BLOCK-START
def optimize_function(x):
    return x*2
#EVOLVE-BLOCK-END

if __name__ =="__main__":
    result = optimize_function(5)
    print(f"result:{result}")