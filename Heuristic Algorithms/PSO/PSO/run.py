import functions
from enumFunctions import Functions
from PSO import PSO

def pso():
    obj_func = functions.selectFunction(Functions.schwefel)
    # dim array size, -5 lb +5 lb 
    PSO(obj_func, -500, 500, 30, 500, 500)

def main():
    pso()

if __name__ == "__main__":
    main()