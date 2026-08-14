import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def full_loss(self, A, prediction, target, margin_losses_per_epoch, args):
        '''
        the total loss function in Equation (9)
        '''
        # the dynamic disambiguation loss in Equation (6):
        Y_candiate = torch.zeros(target.shape).to(device)
        Y_candiate[target > 0] = 1
        prediction_can = prediction * Y_candiate
        new_prediction = prediction_can / prediction_can.sum(dim=1).repeat(prediction_can.size(1), 1).transpose(0, 1)
        d_loss = - torch.sum(target * torch.log(prediction))
        # d_loss = - target * torch.log(prediction)

        # the margin-compliant loss in Equation (10):
        prediction_non = prediction - prediction_can  
        can_p_top1 = torch.max(prediction_can)
        non_p_top1 = torch.max(prediction_non)
        margin_loss = args.mar_scale * torch.pow(1. - can_p_top1 + non_p_top1, 1.).reshape(-1).to(device)
        margin_losses_per_epoch = torch.cat((margin_losses_per_epoch, margin_loss))
        margin_loss_mean = margin_losses_per_epoch.mean(dim=0).requires_grad_(True)  
        margin_loss_std = margin_losses_per_epoch.std(dim=0, unbiased=True).requires_grad_(True)
        m_loss = margin_loss_mean / (1. - margin_loss_std) 

        loss =  d_loss + args.w_lambda * m_loss      # Equation (11)
        # loss =  torch.sum(d_loss) + args.w_lambda * m_loss      # Equation (11)

        return new_prediction, loss, margin_losses_per_epoch
    

def calculate_objective(self, X, Y, margin_losses_per_epoch, args):
    '''
    calculate the full loss
    '''
    Y = Y.reshape(-1)
    Y_logits, A = self.forward(X, args)
    Y_logits = torch.clamp(Y_logits, min=1e-5, max=1.-1e-5)
    Y_prob = F.softmax(Y_logits, dim=1)
    new_prob, loss, margin_losses_per_epoch = self.full_loss(A, Y_prob, Y, margin_losses_per_epoch, args)

    return loss, new_prob, margin_losses_per_epoch