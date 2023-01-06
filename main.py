import torch
import torch.optim as optim
from model.optim import Frobenius_SGD
import argparse
from model.moe_synthetic import MoE
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.manual_seed(111)

def read_data(data_path, label_path):
    x_data = torch.load(data_path)
    x_data = torch.flatten(x_data, start_dim=1)
    x_data = x_data.unsqueeze(dim=1)
    y_data = torch.load(label_path)
    y_data = torch.where(y_data > 0, y_data, 0.)

    return (x_data.cpu(), y_data.cpu())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-train_data_path", default="./dataset/s1_data/train_data.pt")
    parser.add_argument("-train_label_path", default="./dataset/s1_data/train_labels.pt")

    parser.add_argument("-test_data_path", default="./dataset/s1_data/test_data.pt")
    parser.add_argument("-test_label_path", default="./dataset/s1_data/test_labels.pt")

    parser.add_argument("-n_epoch", default=500)    
    parser.add_argument("-lr_global", default=0.001)
    parser.add_argument("-lr_router", default=0.1)
    parser.add_argument("-init_scale", default=1)

    parser.add_argument("-num_patches", default=4)
    parser.add_argument("-num_expert", default=8)
    parser.add_argument("-patch_dim", default=50)
    parser.add_argument("-filter_size", default=16)
    parser.add_argument("-linear", default=True)

    parser.add_argument("-tensorboard_path", default="./runs/synthetic_moe_linear")

    opt =  parser.parse_args()

    writer = SummaryWriter(opt.tensorboard_path)

    train_data = read_data(opt.train_data_path, opt.train_label_path)
    test_data = read_data(opt.test_data_path, opt.test_label_path)

    model = MoE(opt.num_patches, opt.num_expert, opt.patch_dim, opt.filter_size, device, opt.linear, strategy='top-1')
    optimizer_1 = optim.SGD(model.expert.parameters(), lr=opt.lr_global)
    optimizer_2 = Frobenius_SGD(model.gating_param.parameters(), lr=opt.lr_router)
    criterion = torch.nn.CrossEntropyLoss()

    train(opt.n_epoch, train_data, test_data, model, optimizer_1, optimizer_2, criterion, writer, device)
    
def entropy(dispatch): 
    
    return 

def train(n_epoch, train_data, test_data, model, optimizer_1, optimizer_2, criterion, writer, device):
    if device != 'cpu':
        train_data = (train_data[0].to(device), train_data[1].to(device))
        test_data = (test_data[0].to(device), test_data[1].to(device))
        model.to(device)

    # train loop
    for epoch in range(1, n_epoch+1):
        model.train()
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        output, dispatch, load_balancing_loss = model(train_data[0])
        train_loss = criterion(output, train_data[1].type(torch.LongTensor).to(device)) #+ load_balancing_loss * 0.001

        prediction = torch.max(output, dim=-1)
        correct = prediction.indices.eq(train_data[1].type(torch.LongTensor).to(device)).sum()

        train_loss.backward()
        optimizer_1.step()
        optimizer_2.step()

        model.eval()
        with torch.no_grad():
            output, _, _ = model(test_data[0])
            test_loss = criterion(output, test_data[1].type(torch.LongTensor).to(device))
            prediction = torch.max(output, dim=-1)
            test_correct = prediction.indices.eq(test_data[1].type(torch.LongTensor).to(device)).sum()

        print('### EPOCH: %d  ### Train Loss: %.3f  ### Train Accuracy %.3f  ### Test Loss %.3f  ### Test Accuracy %.3f' % (epoch, train_loss.item(), correct/train_data[1].shape[0], test_loss.item(), test_correct/test_data[1].shape[0]))
        writer.add_scalar('Train Loss', train_loss.item(), epoch)
        writer.add_scalar('Test Loss', test_loss.item(), epoch)
        writer.add_scalar('Accuracy', correct/train_data[1].shape[0], epoch)
        writer.add_scalar('Test Accuracy', test_correct/test_data[1].shape[0], epoch)

if __name__ == "__main__":
    main()