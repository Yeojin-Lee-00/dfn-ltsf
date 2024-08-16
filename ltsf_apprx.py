from data_factory import data_provider
from exp_basic import Exp_Basic
import DLinear
from tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from metrics import metric
from func_prune import get_topk_funcs, get_top_func_list_model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time
# get dataloader
from torch.utils.data import DataLoader, Dataset
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
from tools import compute_mape , compute_smape
warnings.filterwarnings('ignore')
from functions import FuncPool, FUNC_CLASSES
from func_manage import transfer_methods

class Exp_Apprx(Exp_Basic):
    def __init__(self, args):
        super(Exp_Apprx, self).__init__(args)
        self.apprx_target = args.apprx_target
        self.func_transfer = args.func_transfer
        self.n_funcs = args.n_func

    def _build_model(self):
        model_dict = {
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            'DLinear': DLinear,
            # 'NLinear': NLinear,
            # 'Linear': Linear,
        }
        
        if self.args.func_transfer == True:
        
            class_keys = list(FUNC_CLASSES.keys())
            classes = [FUNC_CLASSES[key] for key in class_keys]
            # get initial func pool method list
            flist = [dir(FuncPool)[i] for i in range(len(dir(FuncPool))) if not dir(FuncPool)[i].startswith('_')]
            print("Initial len: ", len(flist))
            print("Initial flist: ", flist)
            
            # transfer methods from classes to FuncPool
            transfer_methods(classes, FuncPool)
            # get final func pool method list
            flist = [dir(FuncPool)[i] for i in range(len(dir(FuncPool))) if not dir(FuncPool)[i].startswith('_')]
            print("Transfer_res : ", len(flist))
            print("Transfered flist: ", flist)

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            # save at last epoch
            if epoch == self.args.train_epochs - 1:
                self.model.save_switch = 1

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting +f"transfer_{self.func_transfer}" '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model.alternative_forward(batch_x,self.model.alphas)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pred = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pred, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting +f"transfer_{self.func_transfer}"+ '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('######################mse:{}, mae:{}########################'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        # save mse and mae to csv
        df = pd.DataFrame(columns=['mse', 'mae'])
        new_row = pd.DataFrame({'mse': mse, 'mae': mae}, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(folder_path + 'metrics.csv', index=False, mode='a', header=False)

        return

    def approximate(self, setting):
        # exclude the blank space
        self.apprx_target = self.apprx_target.strip()

        mid_data = self.model.training_datas
        mid_outputs = self.model.training_outputs
        criterion = self._select_criterion()
        # don't need to batchify, as the data is already batchified

        folder_path = './results_apprx/'+ f'{self.apprx_target}/'+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        apprx_epochs = self.args.apprx_epochs
        apprx_optim = optim.Adam(self.model.parameters(), lr=0.01)
        lr_scheduler = optim.lr_scheduler.StepLR(apprx_optim, step_size=150000, gamma=0.1)

        pbar = tqdm(range(apprx_epochs))
        df = pd.DataFrame(columns=['loss','mapes','epoch'])
        
        mid_output_list = []
        output_list = []

        for epoch in pbar:
            epoch_losses = []
            epoch_mapes = []

            if self.args.n_func == 1:
                if epoch == 0 or (epoch+1) % 50 == 0:
                    self.visualize_current_beta(self.model.apprx_seasonal.beta, epoch)
                    self.save_alpha_check(folder_path)
                    self.model.apprx_seasonal.epochs_now = epoch
                    
            stds = []

            temp_output = 0
            temp_mid_output = 0
            last_epoch_output_list = []
            last_epoch_mid_output_list = []

            for data,output in tqdm(zip(mid_data,mid_outputs),total=len(mid_data),leave=False):
                

                data = torch.tensor(data).float().to(self.device)
                output = torch.tensor(output).float().to(self.device)
                
                apprx_out = self.model.apprx_forward(data,self.model.alphas)
                loss = criterion(apprx_out,output) 
                # # # std
                # beta_std = self._get_std(self.model.apprx_seasonal.beta)
                # loss += beta_std
                # stds.append(beta_std.item())

                if epoch==199:
                    last_epoch_output_list.append(torch.norm(output).item())
                    last_epoch_mid_output_list.append(torch.norm(apprx_out).item())

                temp_output += torch.norm(output)
                temp_mid_output += torch.norm(apprx_out)

                apprx_optim.zero_grad()
                loss.backward()
                apprx_optim.step()
                epoch_losses.append(loss.item())
                # mapes
                apprx_out = apprx_out.detach().cpu().numpy()
                output = output.detach().cpu().numpy()
                epoch_mapes.append(compute_smape(apprx_out,output))
                lr_scheduler.step()
            
            mid_output_list.append((temp_mid_output/len(mid_data)).item())
            output_list.append((temp_output/len(mid_data)).item())

            # self.beta_check(folder_path)
            

            pbar.set_postfix({'loss':np.average(epoch_losses),'mapes':np.average(epoch_mapes),'lr':lr_scheduler.get_last_lr(),'beta_stds':np.average(stds)})
            new_row = pd.DataFrame({'loss':np.average(epoch_losses),'mapes':np.average(epoch_mapes),'epoch':epoch},index=[0])
            df = pd.concat([df,new_row],ignore_index=True)
            self.get_alphasum()

        print(f"mid output의 norm의 epoch별 평균: {mid_output_list}")
        print(" ")
        print(f"output의 norm의 epoch별 평균: {output_list}")
        print(" ")
        print(f"last_epoch_output_list: {last_epoch_output_list} \n length: {len(last_epoch_output_list)}")
        print(" ")
        print(f"last epoch_mid_output_list: {last_epoch_mid_output_list} \n length: {len(last_epoch_mid_output_list)}") 
        print(" ")


        
        
        df.to_csv(folder_path + 'result.csv',index=False)
        print(f"Approximation Done,result saved in {folder_path}")
        
        # free up memory
        mid_data = None
        mid_outputs = None
        self.model.training_datas = None 
        self.model.training_outputs = None
        top_func_idx_list = get_topk_funcs(self.model.alphas,3)
        top_func_name_list = get_top_func_list_model(self.model, top_func_idx_list)
        print(f"Top functions are {top_func_name_list}")
        # save the top functions as pandas csv
        df = pd.DataFrame([top_func_name_list])
        df.to_csv(folder_path + 'top_funcs.csv',index=False,mode='a',header=False)
        # save the final alphas 
        alphas = [x.detach().cpu().numpy() for x in self.model.alphas]
        df = pd.DataFrame(alphas[0])
        df.to_csv(folder_path + 'final_alphas.csv',index=False,mode='a',header=False)
    
    def beta_magnitude(self):
        # get the magnitude of beta values, return the magnitude norm.
        beta_norm = []
        for i in range(len(self.model.apprx_seasonal.beta)):
            beta_norm.append(torch.norm(self.model.apprx_seasonal.beta[i]).item())
        return np.average(beta_norm)

    def _get_std(self,beta):
        # get the std of the beta values, return the std.
        beta_std = []
        for i in range(len(beta)):
            beta_std.append(torch.std(beta[i]))
        stds = torch.stack(beta_std)
        norm = torch.norm(stds)
        return norm

    def get_alphasum(self):
        # get the sum of the alpha values, return the sum.
        alpha_sum = []
        for i in range(len(self.model.alphas)):
            alpha_sum.append(torch.sum(self.model.alphas[i]).item())
        import pandas as pd
        df = pd.DataFrame([alpha_sum])
        folder_path = './results_apprx/alphasum/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df.to_csv(folder_path + 'alpha_sum.csv',index=False,mode='a',header=False)

        return
    
    def visualize_current_beta(self, beta,epoch):
        # visualize the heatmap of the beta values, in subplots
        fig,ax = plt.subplots(len(beta),1,figsize=(10,10))
        funcnames = [x for x in dir(FuncPool) if not x.startswith('_')]
        # for i, beta_for_function in enumerate(beta):
        #     beta_for_function = beta_for_function.cpu().detach().numpy()
        #     plt.imshow(beta_for_function)
        #     # plt.colorbar()
        #     plt.tight_layout()
        #     plt.savefig(f'beta_heatmaps/beta_heatmap_{epoch}_func_{i}.png')
        #     plt.clf()
        fig.tight_layout()
        for i, beta_for_function in enumerate(beta):
            beta_for_function = beta_for_function.cpu().detach().numpy()
            ax[i].imshow(beta_for_function)
            ax[i].set_title(f'{funcnames[i]}, alpha: {self.model.alphas[0][0][i].item():.4f}')
        plt.tight_layout()
        plt.text(0.5,0.5,f'Epoch: {epoch}',fontsize=20, color='red')
        plt.savefig(f'beta_heatmaps/reg_beta_heatmap_{epoch}.png')
        plt.clf()
    


    def beta_check(self,path):
        # function by function, get the max beta value.
        max_beta  = []
        for i in range(len(self.model.apprx_seasonal.beta)):
            max_beta.append(torch.max(self.model.apprx_seasonal.beta[i]).item())
        
        # we will consider this as a pandas new row.
        df = pd.DataFrame([max_beta])
        folder_path = path
        df.to_csv(folder_path + 'max_beta.csv',index=False,mode='a',header=False)

    def save_alpha_check(self,path):
        # function by function, get the max beta value.
        # value only.
        df = pd.DataFrame([x.detach().cpu().numpy() for x in self.model.alphas][0])
        folder_path = path
        df.to_csv(folder_path + 'alpha_flows_50.csv',index=False,mode='a',header=False)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
    



        
