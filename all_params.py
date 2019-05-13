# coding=utf-8
import dhsNet

net = dhsNet.MyNet()
params = list(net.parameters())
k = 0
for i in params:
    l = 1
    print("shape：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("params：" + str(l))
    k = k + l
print("all params：" + str(k))



# Model         Model size(MB)      Parameters(m)
# AlexNet       >200                60
# VGG16         >500                138
# GoogleNet     ~50                 6.8
# Inception-v3  90-100              23.3
# dhsNet        12.5                3