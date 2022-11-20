import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('log.csv').values

plt.figure(figsize=(19.2, 10.8))
title='pgd_inf-pgd_2(150-50)'
plt.suptitle(title)

plt.subplot(2,2,1)
plt.plot(list(data[:,0]),list(data[:,1]),color='r',label='clean')
plt.plot(list(data[:,0]),list(data[:,2]),color='g',label='pgd_inf')
plt.plot(list(data[:,0]),list(data[:,3]),color='b',label='pgd_2')
plt.title("test acc")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(list(data[:,0]),list(data[:,7]),color='r',label='clean')
plt.plot(list(data[:,0]),list(data[:,8]),color='g',label='pgd_inf')
plt.plot(list(data[:,0]),list(data[:,9]),color='b',label='pgd_2')
plt.title("train acc")
plt.legend()

plt.subplot(2,2,3)
plt.plot(list(data[:,0]),list(data[:,4]),color='r',label='clean')
plt.plot(list(data[:,0]),list(data[:,5]),color='g',label='pgd_inf')
plt.plot(list(data[:,0]),list(data[:,6]),color='b',label='pgd_2')
plt.title("test loss")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(list(data[:,0]),list(data[:,10]),color='r',label='clean')
plt.plot(list(data[:,0]),list(data[:,11]),color='g',label='pgd_inf')
plt.plot(list(data[:,0]),list(data[:,12]),color='b',label='pgd_2')
plt.title("train loss")
plt.legend()

plt.savefig(title,dpi=300)
plt.close()
