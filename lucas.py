#Iterative function
def lucas(n) :
 
    # declaring base values
    # for positions 0 and 1
    a = 2
    b = 1
     
    if (n == 0) :
        return a
  
    # generating number
    for i in range(2, n + 1) :
        c = a + b
        a = b
        b = c
     
    return b

for i in range(1, 13):
    num =lucas(i)
    print(num)
