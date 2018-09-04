import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)

def exercise_4():
    def taylor_exp(x):
        oldsum=0
        newsum=1
        term=1
        n=0
        while newsum != oldsum:
            n=n+1
            term = term*x/n
            oldsum=newsum
            newsum=newsum+term
        print(n)
        return newsum
    for x in [1,5,10,20,-20]:
        print([x,taylor_exp(x),np.exp(x),np.abs(np.exp(x)-taylor_exp(x))])

def taylor_exp(x):
        oldsum=0
        newsum=1
        term=1
        n=0
        while newsum != oldsum:
            n=n+1
            term = term*(-x)/n
            oldsum=newsum
            newsum=newsum+term
        return 1/newsum
    
    
def exercise_5_b():
    x=np.pi/3
    h=np.flip(np.logspace(-16,-1,16),0)
    
    Df = (np.sin(x+h)-np.sin(x))/h
    
    print(np.transpose([h,Df,0.5-Df]))
    

#exercise_5_b()