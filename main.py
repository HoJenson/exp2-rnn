from train_model import Lab3Model

lrd = True
hidden_size = 128
num_layers = 2
p = 0.1
bidirectional = True

model = Lab3Model(batch_size=64, num_workers=4, path='data/yelp.csv')
model.train(lr=0.01, epochs=80, device='cuda', wait=8, lrd=lrd, hidden_size=hidden_size,
            num_layers=num_layers, p=p, bidirectional=bidirectional)
print('Test score:', model.test())