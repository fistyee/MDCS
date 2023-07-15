import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

eps = 1e-7
import torch.distributed as dist


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):  # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True,
                 reweight_epoch=-1,
                 base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0

            if reweight_epoch != -1:
                idx = 1  # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float,
                                                            requires_grad=False)  # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor  # Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(
                per_cls_weights)  # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                                  requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)

            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature

            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)

            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist,
                                                                                                      mean_output_dist,
                                                                                                      reduction='batchmean')

        return loss


def dkd_loss(logits_student, logits_teacher, target, alpha=1, beta=8, temperature=4):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # pdb.set_trace()

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
            * (temperature ** 2)
        # / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
            * (temperature ** 2)
        # / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return t2



class DiverseLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) #/ np.sum(cls_num_list)

        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = 2

        self.additional_diversity_factor = -0.2
        out_dim = 100
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center1", torch.zeros(1, out_dim))
        self.center_momentum = 0.9
        self.warmup = 20  
        self.reweight_epoch = 200
        if self.reweight_epoch != -1:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float,
                                                        requires_grad=False)  # 这个是logits时算CE loss的weight
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                              requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

    def inverse_prior(self, prior):
        
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0] - 1 - idx1  # reverse the order
        inverse_prior = value.index_select(0, idx2)

        return inverse_prior

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        temperature_mean = 1
        temperature = 1  
        # Obtain logits from each expert
        epoch = extra_info['epoch']
        num = int(target.shape[0] / 2)

        expert1_logits = extra_info['logits'][0] + torch.log(torch.pow(self.prior, -0.5) + 1e-9)      #head

        expert2_logits = extra_info['logits'][1] + torch.log(torch.pow(self.prior, 1) + 1e-9)  # medium

        expert3_logits = extra_info['logits'][2] + torch.log(torch.pow(self.prior, 2.5) + 1e-9)     #few



        teacher_expert1_logits = expert1_logits[:num, :]  # view1
        student_expert1_logits = expert1_logits[num:, :]  # view2

        teacher_expert2_logits = expert2_logits[:num, :]  # view1
        student_expert2_logits = expert2_logits[num:, :]  # view2

        teacher_expert3_logits = expert3_logits[:num, :]  # view1
        student_expert3_logits = expert3_logits[num:, :]  # view2




        teacher_expert1_softmax = F.softmax((teacher_expert1_logits) / temperature, dim=1).detach()
        student_expert1_softmax = F.log_softmax(student_expert1_logits / temperature, dim=1)

        teacher_expert2_softmax = F.softmax((teacher_expert2_logits) / temperature, dim=1).detach()
        student_expert2_softmax = F.log_softmax(student_expert2_logits / temperature, dim=1)

        teacher_expert3_softmax = F.softmax((teacher_expert3_logits) / temperature, dim=1).detach()
        student_expert3_softmax = F.log_softmax(student_expert3_logits / temperature, dim=1)


         

        teacher1_max, teacher1_index = torch.max(F.softmax((teacher_expert1_logits), dim=1).detach(), dim=1)
        student1_max, student1_index = torch.max(F.softmax((student_expert1_logits), dim=1).detach(), dim=1)

        teacher2_max, teacher2_index = torch.max(F.softmax((teacher_expert2_logits), dim=1).detach(), dim=1)
        student2_max, student2_index = torch.max(F.softmax((student_expert2_logits), dim=1).detach(), dim=1)

        teacher3_max, teacher3_index = torch.max(F.softmax((teacher_expert3_logits), dim=1).detach(), dim=1)
        student3_max, student3_index = torch.max(F.softmax((student_expert3_logits), dim=1).detach(), dim=1)


        # distillation
        partial_target = target[:num]
        kl_loss = 0
        if torch.sum((teacher1_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert1_softmax[(teacher1_index == partial_target)],
                                         teacher_expert1_softmax[(teacher1_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher2_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert2_softmax[(teacher2_index == partial_target)],
                                         teacher_expert2_softmax[(teacher2_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher3_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert3_softmax[(teacher3_index == partial_target)],
                                         teacher_expert3_softmax[(teacher3_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        loss = loss + 0.6 * kl_loss * min(extra_info['epoch'] / self.warmup, 1.0)



        # expert 1
        loss += self.base_loss(expert1_logits, target)

        # expert 2
        loss += self.base_loss(expert2_logits, target)

        # expert 3
        loss += self.base_loss(expert3_logits, target)


        return loss

    @torch.no_grad()
    def update_center(self, center, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output))  # * dist.get_world_size())

        # ema update

        return center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536, ncrops=4, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=10, nepochs=200, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
