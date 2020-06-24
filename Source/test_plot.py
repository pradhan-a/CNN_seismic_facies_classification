import matplotlib.pyplot as plt

print ('please, show my graph')


plt.subplot(211)
plt.plot([1,2,3], [1,2,3])
#plt.plot(range(12))
plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background
plt.plot([1,2,3], [1,2,3])
plt.show()