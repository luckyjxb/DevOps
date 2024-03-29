from numpy import genfromtxt,zeros
from pylab import plot,show
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)
print data
print target

print data.shape
print data.shape

s = set(target)
print s

plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
# show()


from pylab import figure, subplot, hist, xlim, show
xmin = min(data[:,0])
xmax = max(data[:,0])
figure()
subplot(411) # distribution of the setosa class (1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
# show()


t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(data,t) # training on the iris dataset

print classifier.predict(data[0])