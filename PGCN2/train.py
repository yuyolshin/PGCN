import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')

# PeMS-Bay
parser.add_argument('--study_area', type=str, default='SF', help='study area')
parser.add_argument('--num_nodes', type=int, default=325, help='number of nodes')
parser.add_argument('--data', type=str, default='/home/user/YS/PGCN/data/PEMS-BAY',
                    help='data path')
parser.add_argument('--adjdata', type=str, default='/home/user/YS/PGCN/data/sensor_graph/adj_mx_bay.pkl',
                    help='adj data path')
parser.add_argument('--save', type=str, default='/home/user/YS/PGCN/PGCN2/dyn_only/PeMS-Bay-bs32/',
                    help='save path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')

# # METR-LA
# parser.add_argument('--study_area', type=str, default='LA', help='study area')
# parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
# parser.add_argument('--data', type=str, default='/home/user/YS/PGCN/data/METR-LA',
#                     help='data path')
# parser.add_argument('--adjdata', type=str, default='/home/user/YS/PGCN/data/sensor_graph/adj_mx.pkl',
#                     help='adj data path')
# parser.add_argument('--save', type=str, default='/home/user/YS/PGCN/PGCN2/dyn_only/METR-LA-bs32/',
#                     help='save path')
# parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')

# # Urban-core
# parser.add_argument('--study_area', type=str, default='Seoul', help='study area')
# parser.add_argument('--num_nodes', type=int, default=304, help='number of nodes')
# parser.add_argument('--data', type=str, default="/home/user/YS/PGCN/data/urban-core",
#                     help='data path')
# parser.add_argument('--adjdata1',
#                     default='/home/user/YS/PGCN/data/urban-core-adj/outADJ1_dist_og.csv',
#                     help='out adj')
# parser.add_argument('--adjdata2',
#                     default='/home/user/YS/PGCN/data/urban-core-adj/inADJ1_dist_og.csv',
#                     help='in adj')
# parser.add_argument('--save', type=str, default='/home/user/YS/PGCN/PGCN2/dyn_only/Urban-core-bs32/',
#                     help='save path')
# parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')

# # # Seattle Loop
# parser.add_argument('--study_area', type=str, default='Seattle', help='study area')
# parser.add_argument('--num_nodes', type=int, default=323, help='number of nodes')
# parser.add_argument('--data', type=str, default="/home/user/YS/PGCN/data/seattleLoop/LOOP",
#                     help='data path')
# parser.add_argument('--adjdata',
#                     default='/home/user/YS/PGCN/data/seattleLoop/Loop_Seattle_2015_A.npy',
#                     help='out adj')
# parser.add_argument('--save', type=str, default='/home/user/YS/PGCN/PGCN2/dynAdp/Seattle-loop-bs32/',
#                     help='save path')
# parser.add_argument('--adjtype', type=str, default='symnadj', help='adj type')

parser.add_argument('--gcn_bool', action='store_false', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_false', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_false', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed', type=int, default=99, help='random seed')
# parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main(expid):
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype, args.stduy_area)
    if args.study_area == 'Seoul':
        adj_mx = util.load_adj([args.adjdata1, args.adjdata2], args.adjtype, args.study_area)
    elif args.study_area == 'Seattle':
        adj_mx = util.load_adj(args.adjdata, args.adjtype, args.study_area)
    else:
        adj_mx = util.load_adj(args.adjdata, args.adjtype, args.study_area)
    dataloader, adp = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    adp = torch.Tensor(adp).to(device)
    print(adp.size())
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, adp=adp)
    print(sum(p.numel() for p in engine.model.parameters() if p.requires_grad))
    sys.exit()
    # """
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "exp_" + str(expid) + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(expid) + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    # engine.model.load_state_dict(
    #     torch.load(args.save + "_epoch_92_2.21.pth"))

    # engine.model.load_state_dict(
    #     torch.load('/media/hdd1/YS/KDD22/newModel/adp/PeMS-Bay/best/best_epoch_95_1.61.pth'))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    print('best epoch: ', str(bestid))
    torch.save(engine.model.state_dict(),
               args.save + 'best/exp_' + str(expid) + '_best_epoch_' + str(bestid) +
               '_' + str(round(his_loss[bestid], 2)) + ".pth")

    """

    # engine.model.load_state_dict(
    #     torch.load('/media/hdd1/YS/KDD22/newModel/dynOnly/PeMS-Bay/best/best_epoch_93_1.89.pth'))
    # engine.model.load_state_dict(
    #     torch.load('/media/hdd1/YS/KDD22/newModel/dynAdp/PeMS-Bay/best/best_epoch_86_1.61.pth'))
    engine.model.load_state_dict(
        torch.load('/media/hdd1/YS/KDD22/newModel/dynAdpOnly/METR-LA/best/best_epoch_95_2.95.pth'))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    engine.model.eval()

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        # if iter == 0:
        #     print(testx[:5, 0, 0, :5])
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        # if iter == 0:
        #     print(preds[:5, 0, 0, :5])
        outputs.append(preds.squeeze(1))

    yhat1 = torch.cat(outputs, dim=0)
    yhat1 = yhat1[:realy.size(0), ...]
    # print(realy[:5, 0, :5])
    # print(yhat1[:5, 0, :5])

    del engine, outputs, testx, preds

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, adp=adp)

    # engine.model.load_state_dict(
    #     torch.load('/media/hdd1/YS/KDD22/newModel/dynOnly/PeMS-Bay/best/best_epoch_94_1.91.pth'))
    engine.model.load_state_dict(
        torch.load('/media/hdd1/YS/KDD22/newModel/dynAdpOnly/METR-LA/best/best_epoch_97_3.0.pth'))

    outputs = []

    engine.model.eval()

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))

    yhat2 = torch.cat(outputs, dim=0)
    yhat2 = yhat2[:realy.size(0), ...]

    del engine, outputs, testx, preds

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit, adp=adp)

    # engine.model.load_state_dict(
    #     torch.load('/media/hdd1/YS/KDD22/newModel/dynAdp/Urban-core/best/best_epoch_95_2.9.pth'))
    # engine.model.load_state_dict(
    #     torch.load('/media/hdd1/YS/KDD22/newModel/dynOnly/METR-LA/best/best_epoch_98_2.79.pth'))

    outputs = []

    engine.model.eval()

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))

    yhat3 = torch.cat(outputs, dim=0)
    yhat3 = yhat3[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred1 = scaler.inverse_transform(yhat1[:, :, i])
        pred2 = scaler.inverse_transform(yhat2[:, :, i])
        pred3 = scaler.inverse_transform(yhat3[:, :, i])
        real = realy[:, :, i]
        metrics1 = util.metric(pred1, real)
        metrics2 = util.metric(pred2, real)
        metrics3 = util.metric(pred3, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        tmpMAE = (metrics1[0] + metrics2[0] + metrics3[0]) / 3
        tmpMAPE = (metrics1[1] + metrics2[1] + metrics3[1]) / 3 * 100
        tmpRMSE = (metrics1[2] + metrics2[2] + metrics3[2]) / 3
        # tmpMAE = (metrics1[0] + metrics2[0]) / 2
        # tmpMAPE = (metrics1[1] + metrics2[1]) / 2 * 100
        # tmpRMSE = (metrics1[2] + metrics2[2]) / 2
        # tmpMAE = metrics1[0]
        # tmpMAPE = metrics1[1] * 100
        # tmpRMSE = metrics1[2]
        if (i == 2) | (i == 5) | (i == 11):
            # print(log.format(i + 1, metrics1[0], metrics1[2], metrics1[1]))
            # print(log.format(i + 1, metrics2[0], metrics2[2], metrics2[1]))
            # print(log.format(i + 1, metrics3[0], metrics3[2], metrics3[1]))
            print(log.format(i + 1, tmpMAE, tmpRMSE, tmpMAPE))
        # amae.append(metrics[0])
        # amape.append(metrics[1])
        # armse.append(metrics[2])

    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    """


if __name__ == "__main__":
    t1 = time.time()
    # for i in range(3):
    main(0)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
