import torch
import torch.optim as optim
from model.optim import Frobenius_SGD
import argparse
from model.moe_synthetic import MoE


def read_data(data_path, label_path):
    x_data = torch.load(data_path)
    x_data = torch.flatten(x_data, start_dim=1)
    x_data = x_data.unsqueeze(dim=1)
    y_data = torch.load(label_path)
    y_data = torch.where(y_data > 0, y_data, 0.)

    return (x_data.cpu(), y_data.cpu())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-train_data_path", default="./dataset/cluster_all_data.pt")
    parser.add_argument("-train_label_path", default="./dataset/cluster_all_data_label.pt")

    parser.add_argument("-test_data_path", default="./MoE_Code4/synthetic_data_s2/test_data.pt")
    parser.add_argument("-test_label_path", default="./MoE_Code4/synthetic_data_s2/test_labels.pt")

    parser.add_argument("-n_epoch", default=500)    
    parser.add_argument("-lr_global", default=0.001)
    parser.add_argument("-lr_router", default=0.1)
    parser.add_argument("-init_scale", default=1)

    parser.add_argument("-num_patches", default=4)
    parser.add_argument("-num_expert", default=1)
    parser.add_argument("-patch_dim", default=50)
    parser.add_argument("-filter_size", default=512)
    parser.add_argument("-linear", default=False)

    opt =  parser.parse_args()

    train_data = read_data(opt.train_data_path, opt.train_label_path)
    test_data = read_data(opt.test_data_path, opt.test_label_path)

    model = MoE(opt.num_patches, opt.num_expert, opt.patch_dim, opt.filter_size, opt.linear, strategy='top-1')
    optimizer_1 = optim.SGD(model.parameters(), lr=opt.lr_global)
    optimizer_2 = Frobenius_SGD(model.gating_param.parameters(), lr=opt.lr_router)
    criterion = torch.nn.CrossEntropyLoss()

    train(opt.n_epoch, train_data, test_data, model, optimizer_1, optimizer_2, criterion)
    
def entropy(dispatch): 
    
    return 

def train(n_epoch, data, test_data, model, optimizer_1, optimizer_2, criterion):
    # train loop
    for epoch in range(0, n_epoch):
        model.train()
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        output, dispatch, load_balancing_loss = model(data[0])
        loss = criterion(output, data[1].type(torch.LongTensor)) #+ load_balancing_loss * 0.001

        prediction = torch.max(output, dim=-1)
        correct = prediction.indices.eq(data[1].type(torch.LongTensor)).sum()

        loss.backward()
        optimizer_1.step()
        optimizer_2.step()

        model.eval()
        with torch.no_grad():
            output, _, _ = model(test_data[0])
            prediction = torch.max(output, dim=-1)
            test_correct = prediction.indices.eq(test_data[1].type(torch.LongTensor)).sum()

        print('### EPOCH: %d  ### Loss: %.3f  ### Accuracy %.3f  ### Test Accuracy %.3f' % (epoch, loss.item(), correct/data[1].shape[0], test_correct/test_data[1].shape[0]))

if __name__ == "__main__":
    main()