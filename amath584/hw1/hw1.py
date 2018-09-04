import scipy as sp
    
def exercise_1_1(B):
    
    M=sp.copy(B)
    M[:,0]=2*M[:,0] # double column 1
    M[2]=1/2*M[2] # halve row 3
    M[0]=M[2]+M[0] # add row 3 to row 1
    M[:,[0,3]]=M[:,[3,0]] # interchange columns 1 and 4
    M[[0,2,3]]=M[[0,2,3]]-M[1] # subtract row 2 from each of the other rows
    M[:,3]=M[:,2] # replace column 4 by column 3
    M[:,0]=0 # delete column 1    
    
    O1=sp.matrix([[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    O2=sp.matrix([[1,0,0,0],[0,1,0,0],[0,0,1/2,0],[0,0,0,1]])
    O3=sp.matrix([[1,0,1,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    O4=sp.matrix([[0,0,0,1],[0,1,0,0],[0,0,1,0],[1,0,0,0]])
    O5=sp.matrix([[1,-1,0,0],[0,1,0,0],[0,-1,1,0],[0,-1,0,1]])
    O6=sp.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,0]])
    O7=sp.matrix([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
    A=sp.matrix([[1,-1,1/2,0],[0,1,0,0],[0,-1,1/2,0],[0,-1,0,1]])
    C=sp.matrix([[0,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,0]])

    print(M)
    print(O5*O3*O2*B*O1*O4*O6*O7)
    print(A*B*C)
    
    print(sp.array_equal(M,O5*O3*O2*B*O1*O4*O6*O7) and sp.array_equal(M,A*B*C))

exercise_1_1(sp.matrix(sp.random.rand(4,4)))
