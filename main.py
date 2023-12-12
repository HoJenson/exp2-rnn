from train_model import Lab3Model

# 学习率衰减
lrd = True
# GRU模型中隐藏状态的维度
hidden_size = [64, 128, 256]
# GRU模型的层数
num_layers = [1, 2, 3, 4, 5]
# dropout概率
p = 0.0
# 是否使用双向GRU
bidirectional = True

model = Lab3Model(batch_size=128, num_workers=0, path='data/yelp_m.csv')
model.train(lr=0.01, epochs=80, device='cuda', wait=8, lrd=lrd, hidden_size=hidden_size,
            num_layers=num_layers, p=p, bidirectional=bidirectional, fig_name='lab3')
print('Test score:', model.test())