import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16
import model.model as module_arch
import pdb
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)
        print("self.add_extra_info",self.add_extra_info)

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader )
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader )
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        #self.start_epoch = 60
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        print(len(self.data_loader))
        self.schedule = self.cosine_scheduler(0.996,1,200,len(self.data_loader ))

    def get_agent_weight(self, epoch):
        agent_weights = [0, 1 , 1]
        loss_weights = [0, 1, 1]
        agent_milestones = [0,  60,  80]
        loss_milestones = [0,  60,  80]
        agent_weight = agent_weights[0]
        loss_weight = loss_weights[0]
        for i, ms in enumerate(agent_milestones):
            if epoch >= ms:
                agent_weight = agent_weights[i]
        for i, ms in enumerate(loss_milestones):
            if epoch >= ms:
                loss_weight = loss_weights[i]
        self.logger.info('Center Weight: {}'.format(agent_weight))
        return agent_weight, loss_weight

    def cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule 
   
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        # saving training info for environments building
        all_ind = []
        all_lab = []
        all_prb = []
        all_lgt = []
        agent_weight,loss_weight = self.get_agent_weight(epoch)        
        #train_cls_num_list = np.array(env1_loader.cls_num_list)
        for batch_idx, data in enumerate(self.data_loader):
        #for batch_idx, (data, data2) in enumerate(zip(env1_loader, env2_loader)):
            it = len(self.data_loader) * epoch+batch_idx
            #pdb.set_trace()
            view1, view2, target,  index1 = data
            view1, view2, target  = view1.to(self.device), view2.to(self.device), target.to(self.device) 

            #data2, target2  ,index2 = data2
            #data2, target2  = data2.to(self.device), target2.to(self.device) 
 
            data = torch.cat([view1, view2], dim=0).cuda()
            target = torch.cat([target, target], dim=0).cuda()
 
            #indexs = torch.cat([index1, index2], dim=0).cuda()
           
            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target) 

                    output, loss = output   
                else:
                    extra_info = {}
                    output = self.model(data)
                    #pdb.set_trace()
                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                                   
                            
                            
                            extra_info.update({
                                "loss_weight": loss_weight,
                                "agent_weight": agent_weight,
                                "feat": output["feat"].transpose(0,1),
                                "agent": logits,
                                "logits": logits.transpose(0, 1),
                                "epoch":epoch
                            })
                        else:
                            extra_info.update({
                                "logits": self.real_model.backbone.logits
                            })
                    #pdb.set_trace()  
                    '''
                    with torch.no_grad():
                        #EMA
                        m = self.schedule[it]
                        for param_q, param_k in zip(self.model.backbone.linear_head.parameters(), self.model.backbone.linears[0].parameters()):
                             param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                        for param_q, param_k in zip(self.model.backbone.linear_few.parameters(), self.model.backbone.linears[2].parameters()):
                             param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    ''' 
                    if isinstance(output, dict):
                        output =  output["output"]#output['logits'][:,0,:]#

                    if self.add_extra_info:
                        loss = self.criterion(output_logits=output, target=target, extra_info=extra_info)
                    else:
                        loss = self.criterion(output_logits=output, target=target) 
                    
            
            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            '''
            # save info for environment spliting
            all_lgt.append(output.detach().clone().cpu())
            predictions = output.softmax(-1)
            gt_score = torch.gather(predictions, 1, torch.unsqueeze(target, 1)).view(-1)
            all_ind.append(indexs.detach().clone().cpu())
            all_lab.append(target.detach().clone().cpu())
            all_prb.append(gt_score.detach().clone().cpu())
            '''
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
         
        # save env score
        '''
        env_score_memo = {}
        update_milestones = [ 260,  280]
        if epoch in update_milestones:
                # update env mask
                self.all_ind = torch.cat(all_ind, dim=0)
                self.all_lab = torch.cat(all_lab, dim=0)
                self.all_prb = torch.cat(all_prb, dim=0)
                self.all_lgt = torch.cat(all_lgt, dim=0)

                # save env_score
                env_score_memo['label_{}'.format(epoch)] = self.all_lab.tolist()
                env_score_memo['prob_{}'.format(epoch)] = self.all_prb.tolist()
                env_score_memo['idx_{}'.format(epoch)] = self.all_ind.tolist()

                self.update_env_by_score(env1_loader, env2_loader, total_image)            
        '''   

        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log#s,env1_loader, env2_loader
    
    def update_env_by_score(self, env1_loader, env2_loader, total_image):
        # seperate environments by inter-score + intra-score
        all_ind, all_lab, all_prb = self.all_ind.tolist(), self.all_lab.tolist(), self.all_prb.tolist()
        all_cat = list(set(all_lab))
        all_cat.sort()
        cat_socres = {cat:{} for cat in all_cat}
        all_scores = {}
        for ind, lab, prb in zip(all_ind, all_lab, all_prb):
            cat_socres[lab][ind] = prb
            all_scores[ind] = prb

        
        # baseline distribution
        env1_score = torch.zeros(total_image).fill_(1.0)
        env2_score = torch.zeros(total_image).fill_(1.0)
        # inverse distribution

        
        intra_weight = self.generate_intra_weight(cat_socres, total_image, tg_scale=4.0)
        env2_score = env2_score * intra_weight

        env1_loader.sampler.set_parameter(env1_score)
        env2_loader.sampler.set_parameter(env2_score)

    def generate_inter_weight(self, all_scores, total_image, tg_scale=4.0):
        # normalize
        inter_weight = torch.zeros(total_image).fill_(1.0)
        for ind, prb in all_scores.items():
            inter_weight[ind] = prb
        inter_weight = inter_weight - inter_weight.min()
        inter_weight = inter_weight / (inter_weight.max() + 1e-9)

        # use Pareto principle to determine the scale parameter
        inter_weight = (1.0 - inter_weight).abs() + 1e-5
        head_mean = torch.topk(inter_weight, k=int(total_image * 0.8), largest=False)[0].mean().item()
        tail_mean = torch.topk(inter_weight, k=int(total_image * 0.2), largest=True )[0].mean().item()
        scale = tail_mean / head_mean + 1e-5
        exp_scale = torch.FloatTensor([tg_scale]).log() / torch.FloatTensor([scale]).log()
        exp_scale = exp_scale.clamp(min=1, max=10)
        self.logger.info('Inter Score Original Head (80) Tail (20) Scale is {}'.format(scale))
        self.logger.info('Inter Score Target   Head (80) Tail (20) Scale is {}'.format(tg_scale))
        self.logger.info('Inter Score Exp Scale is {}'.format(exp_scale.item()))
        inter_weight = inter_weight ** exp_scale
        inter_weight = inter_weight + 1e-12
        inter_weight = inter_weight / inter_weight.sum()
        return inter_weight

    def generate_intra_weight(self, cat_socres, total_image, tg_scale=4.0):
        # normalize
        intra_weight = torch.zeros(total_image).fill_(0.0)
        for cat, cat_items in cat_socres.items():
            cat_size = len(cat_items)
            if cat_size < 5:
                for ind in list(cat_items.keys()):
                    intra_weight[ind] = 1.0 / max(cat_size, 1.0)
                continue
            cat_inds = list(cat_items.keys())
            cat_scores = torch.FloatTensor([cat_items[ind] for ind in cat_inds])
            cat_scores = cat_scores - cat_scores.min()
            cat_scores = cat_scores / (cat_scores.max() + 1e-9)

            # use Pareto principle to determine the scale parameter
            cat_scores = (1.0 - cat_scores).abs() + 1e-5
            head_mean = torch.topk(cat_scores, k=int(cat_size * 0.8), largest=False)[0].mean().item()
            tail_mean = torch.topk(cat_scores, k=int(cat_size * 0.2), largest=True )[0].mean().item()
            scale = tail_mean / head_mean + 1e-5
            exp_scale = torch.FloatTensor([tg_scale]).log() / torch.FloatTensor([scale]).log()
            exp_scale = exp_scale.clamp(min=1, max=10)
            if int(cat) == 0:
                self.logger.info('Intra Score at Cat-{} Original Head (80) Tail (20) Scale is {}'.format(cat, scale))
                self.logger.info('Intra Score at Cat-{} Target   Head (80) Tail (20) Scale is {}'.format(cat, tg_scale))
                self.logger.info('Intra Score at Cat-{} Exp Scale is {}'.format(cat, exp_scale.item()))
            cat_scores = cat_scores ** exp_scale
            cat_scores = cat_scores + 1e-12
            cat_scores = cat_scores / cat_scores.sum()
            for ind, score in zip(cat_inds, cat_scores.tolist()):
                intra_weight[ind] = score
        self.logger.info('Intra Total Score {}, which should be equal to NUM_CLASS'.format(intra_weight.sum().item()))
        return intra_weight
        



    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
           
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if isinstance(output, dict):
                    output = output["output"]
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
