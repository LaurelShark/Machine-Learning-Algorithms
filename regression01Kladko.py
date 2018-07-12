from numpy import *
x = array([0,1,2,3,4,5, 30])
y = array([0,0.8,0.9,0.1,-0.8,-1, 5 ])
print(x)
print(y)

from scipy.interpolate import *
p1 = polyfit(x,y,1)
p2 = polyfit(x,y,2)
p3 = polyfit(x,y,3)
print(p1)
print(p2)
print(p3)

from matplotlib.pyplot import *
plot(x,y,'o')
xp = linspace(-2,30,100)
plot(xp,polyval(p1,xp),'r-')
plot(xp,polyval(p2,xp),'b--')
plot(xp,polyval(p3,xp),'m:')
yfit = p1[0] * x + p1[1]
yresid= y - yfit
SSresid = sum(pow(yresid,2))
SStotal = len(y) * var(y)
rsq = 1 - SSresid/SStotal
print('------------')
print(yfit)

print('------------')

print(y)
print(rsq)

from scipy.stats import *
slope,intercept,r_value,p_value,std_err = linregress(x,y)
print(pow(r_value,2))
print(p_value)
show()
