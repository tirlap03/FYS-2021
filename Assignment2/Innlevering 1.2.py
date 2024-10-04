print ("oppgave 1")

T = 7

if (T % 7) == 0:
    print ("delig med 7")
    
else:
    print ("ikke delig")

    


print("oppgave 2")

a =  4-6
b = 6-4


print((a)*(-1))
print((b))





print("oppgave 3) a")
A = 2
B = 4
C = 6 

if A>B:
    print(A)
    
elif B>C:
    print(B)
    
else: print(C)





print("oppgave 3)b")

A,C = C,A
print("A er nå",A)
print("C er nå",C)



print("oppgave 3)c")

numbers = (A,B,C)
print(sorted(numbers))




print("oppgave 4")

for i in range(1,51) :
    if i % 2 == 0:
        partall_oddetall = "partall "
    else:
        partall_oddetall = "oddetall"

if i % 7 == 0 and i != 0:
    delig_med_7 = "delig med 7"
else: 
    delig_med_7 = "ikke delig med 7"
    
print(i, "er", partall_oddetall, "og",delig_med_7)




print("oppgave 5.1")
import math
k = 1
s = 0 

while k <= 40:
    s += math.cos(k) / math.sin(k)
    k += 1
print(f"while løkke= {s:.2f}")




print("oppgave 5.2")
s2 = 0

for k in range(1,41):
    s2 += math.cos(k) / math.sin(k)
print (f"for- løkke= {s2:.2f}") 


