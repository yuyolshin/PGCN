import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--gcn_bool', action='store_false', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_false', help='whether add adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

# PeMS-Bay
parser.add_argument('--study_area', type=str, default='SF', help='study area')
parser.add_argument('--data', type=str, default='/home/user/YS/PGCN/data/PEMS-BAY',
                    help='data path')
parser.add_argument('--adjdata', type=str, default='/home/user/YS/PGCN/data/sensor_graph/adj_mx_bay.pkl',
                    help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--checkpoint',
                    default='/home/user/YS/PGCN/PGCN2/dyn/PeMS-Bay-bs32/best/exp_0_best_epoch_187_1.58.pth',
                    type=str, help='')
args = parser.parse_args()


def main():
    device = torch.device(args.device)

    if args.study_area == 'Seoul':
        adj_mx = util.load_adj([args.adjdata1, args.adjdata2], args.adjtype, args.study_area)
    elif args.study_area == 'Seattle':
        adj_mx = util.load_adj(args.adjdata, args.adjtype, args.study_area)
    else:
        adj_mx = util.load_adj(args.adjdata, args.adjtype, args.study_area)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if args.aptonly:
        supports = None

    dataloader, adp = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    adp = torch.Tensor(adp).to(device)
    scaler = dataloader['scaler']
    outputs = []
    # realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    model = gwnet(device, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('model load successfully')

    # for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]


    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[2], metrics[1] * 100))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    y12 = realy[:, :, 11].cpu().detach().numpy()
    yhat12 = scaler.inverse_transform(yhat[:, :, 11]).cpu().detach().numpy()

    y3 = realy[:, :, 2].cpu().detach().numpy()
    yhat3 = scaler.inverse_transform(yhat[:, :, 2]).cpu().detach().numpy()

    pd.DataFrame(y12).to_csv('/home/user/YS/PGCN/PGCN2/dyn/PeMS-Bay-bs32/best/y12.csv',
                             header=None, index=False)
    pd.DataFrame(yhat12).to_csv('/home/user/YS/PGCN/PGCN2/dyn/PeMS-Bay-bs32/best/yhat12.csv',
                                header=None, index=False)
    pd.DataFrame(y3).to_csv('/home/user/YS/PGCN/PGCN2/dyn/PeMS-Bay-bs32/best/y3.csv',
                            header=None, index=False)
    pd.DataFrame(yhat3).to_csv('/home/user/YS/PGCN/PGCN2/dyn/PeMS-Bay-bs32/best/yhat3.csv',
                               header=None, index=False)

if __name__ == "__main__":
    main()
