import torch
from tqdm import tqdm
from utils.utils import get_lr


def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, scaler = None , data_type = None , use_fp16=False):
    loss = 0
    val_loss = 0
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).to(data_type).cuda()
                    targets = [torch.from_numpy(ann).to(data_type).cuda() for ann in targets]

                else:
                    images = torch.from_numpy(images).to(data_type)
                    targets = [torch.from_numpy(ann).to(data_type) for ann in targets]

            if use_fp16:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model_train(images.half())
            else:
                outputs = model_train(images)

            # print("\n")
            # for x in outputs :
            #     print(x.shape)
            # print("\n")

            loss_value = yolo_loss(outputs, targets)
            if use_fp16:
                optimizer.zero_grad()
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

            loss += loss_value.item()
            # lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            pbar.set_postfix(**{'loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

                optimizer.zero_grad()
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, targets)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(),
               'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
