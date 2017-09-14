import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("tweet.txt","r").read()
    lines = pullData.split('\n')
    
    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines[-200:]:
        x += 1
        print(l.split(" "))
        if "pos" in l:
            y += float(l.split(" ")[1])
        elif "neg" in l:
            y -= float(l.split(" ")[1])
        else:
            y += 1.0
        xar.append(x)
        yar.append(y)
        
    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()