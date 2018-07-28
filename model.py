import torch
import retro
import time
import numpy as np
import torch.multiprocessing as mp
from torch.optim import Adam
from network import ActorCritic
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(48),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))])


class TraceQueue(object):
    """
    single-machine implementation
    """

    def __init__(self):
        # traces: 0-[s, a, r, p]
        #         1-[s, a, r, p]
        # trace(s, a, rew, a_prob)
        # s[n_step+1, 3, width, height]
        # a[n_step, a_space]
        # rew[n_step]
        # a_prob[n_step]
        self.traces_s = []
        self.traces_a = []
        self.traces_r = []
        self.traces_p = []
        self.lock = mp.Lock()

    def push(self, trace):
        if len(self.traces_s) > 400:
            print("queue size({}) is too large, wait...".format(len(self.traces_s)))
            time.sleep(10)
        # in
        with self.lock:
            self.traces_s.append(trace[0])
            self.traces_a.append(trace[1])
            self.traces_r.append(trace[2])
            self.traces_p.append(trace[3])

    def pop(self, num):
        while len(self.traces_s) < num:
            print("queue waiting...{}/{}".format(len(self.traces_s), num))
            time.sleep(1)
        # out
        with self.lock:
            res_s, res_a, res_r, res_p = self.traces_s[:num], self.traces_a[:num], \
                                         self.traces_r[:num], self.traces_p[:num]
            del self.traces_s[:num]
            del self.traces_a[:num]
            del self.traces_r[:num]
            del self.traces_p[:num]

        # stack batch
        return torch.stack(res_s, dim=0), \
               torch.stack(res_a, dim=0), \
               torch.stack(res_r, dim=0), \
               torch.stack(res_p, dim=0)


class Learner(object):
    def __init__(self, opt, trace_queue):
        self.opt = opt
        self.trace_queue = trace_queue
        self.network = ActorCritic(opt).to(device)
        self.optimizer = Adam(self.network.parameters(), lr=opt.lr)
        self.network.share_memory()

    def learning(self):
        torch.manual_seed(self.opt.seed)
        coef_hat = torch.Tensor([[self.opt.coef_hat]])
        rho_hat = torch.Tensor([[self.opt.rho_hat]])
        while True:
            # batch-trace
            # s[batch, n_step+1, 3, width, height]
            # a[batch, n_step, a_space]
            # rew[batch, n_step]
            # a_prob[batch, n_step, a_space]
            s, a, rew, prob = self.trace_queue.pop(self.opt.batch_size)
            ###########################
            # variables we need later #
            ###########################
            v, coef, rho, entropies, log_prob = [], [], [], [], []
            cx = torch.zeros(self.opt.batch_size, 256).to(device)
            hx = torch.zeros(self.opt.batch_size, 256).to(device)
            for step in range(s.size(1)):
                # value[batch]
                # logit[batch, 12]
                value, logit, (hx, cx) = self.network((s[:, step, ...], (hx, cx)))
                v.append(value)
                if step >= a.size(1):  # noted that s[, n_step+1, ...] but a[, n_step,...]
                    break              # loop for n_step+1 because v in n_step+1 is needed.

                # π/μ[batch]
                # TODO: cumprod might produce runtime problem
                logit_a = a[:, step, :] * logit.detach() + (1 - a[:, step, :]) * (1 - logit.detach())
                prob_a = a[:, step, :] * prob[:, step, :] + (1 - a[:, step, :]) * (1 - prob[:, step, :])
                is_rate = torch.cumprod(logit_a/(prob_a + 1e-6), dim=1)[:, -1]
                coef.append(torch.min(coef_hat, is_rate))
                rho.append(torch.min(rho_hat, is_rate))

                # enpy_aspace[batch, 12]
                # calculating the entropy[batch, 1]
                # more specifically there are [a_space] entropy for each batch, sum over them here.
                # noted that ~do not~ use detach here
                enpy_aspace = - torch.log(logit) * logit - torch.log(1-logit) * (1-logit)
                enpy = (enpy_aspace).sum(dim=1, keepdim=True)
                entropies.append(enpy)

                # calculating the prob that the action is taken by target policy
                # and the prob_pi_a[batch, 12] and log_prob[batch, 1] of this action
                # noted that ~do not~ use detach here
                prob_pi_a = (a[:, step, :] * logit) + (1 - a[:, step, :]) * (1 - logit)
                log_prob_pi_a = torch.log(prob_pi_a).sum(dim=1, keepdim=True)
                log_prob.append(log_prob_pi_a)
                # prob_pi_a = torch.cumprod(prob_pi_a, dim=1)[:, -1:]
                # log_prob_pi_a = torch.log(prob_pi_a)

            ####################
            # calculating loss #
            ####################
            policy_loss = 0
            value_loss = 0
            # gae = torch.zeros(self.opt.batch_size, 1)
            for rev_step in reversed(range(s.size(1) - 1)):
                # compute v_(s+1)[batch] for policy gradient
                fix_vp = rew[:, rev_step] + self.opt.gamma * (v[rev_step+1] + value_loss) - v[rev_step]

                # value_loss[batch]
                td = rew[:, rev_step] + self.opt.gamma * v[rev_step + 1] - v[rev_step]
                value_loss = self.opt.gamma * coef[rev_step] * value_loss + rho[rev_step] * td

                # policy_loss = policy_loss - log_probs[i] * Variable(gae)
                # the td must be detach from network-v

                # # dalta_t[batch]
                # delta_t = rew[:, rev_step] + self.opt.gamma * v[rev_step + 1] - v[rev_step]
                # gae = gae * self.opt.gamma + delta_t.detach()

                policy_loss = policy_loss \
                              - rho[rev_step] * log_prob[rev_step] * fix_vp.detach() \
                              - self.opt.entropy_coef * entropies[rev_step]

            self.optimizer.zero_grad()
            policy_loss = policy_loss.sum()
            value_loss = value_loss.sum()
            loss = policy_loss + self.opt.value_loss_coef * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.opt.max_grad_norm)
            print("v_loss {:.3f} p_loss {:.3f}".format(value_loss.item(), policy_loss.item()))
            self.optimizer.step()


class Actor(object):
    def __init__(self, opt, trace_queue, learner):
        self.opt = opt
        self.trace_queue = trace_queue
        self.learner = learner

        # 游戏
        self.env = None
        # s_channel = self.env.observation_space.shape[0]
        # a_space = self.env.action_space

        # 网络
        self.behaviour = ActorCritic(opt)  # .to(device)

    def performing(self, rank):
        torch.manual_seed(self.opt.seed)
        # 每个线程初始化环境
        self.env = retro.make(game=self.opt.env)
        self.env.seed(self.opt.seed + rank)

        s = self.env.reset()
        s = transform(s).unsqueeze(dim=0)  # .to(device)
        episode_length = 0
        r_sum = 0.
        done = True
        while True:
            # apply
            # print(type(self.learner.network.state_dict()))
            self.behaviour.load_state_dict(self.learner.network.state_dict())
            # LSTM
            if done:
                cx = torch.zeros(1, 256)  # .to(device)
                hx = torch.zeros(1, 256)  # .to(device)
            else:
                cx = cx.detach()
                hx = hx.detach()

            trace_s, trace_a, trace_rew, trace_aprob = [], [], [], []
            # collect n-step
            for n in range(self.opt.n_step):
                episode_length += 1
                #  add to trace - 0
                trace_s.append(s)
                value, logit, (hx, cx) = self.behaviour((s, (hx, cx)))
                logit = logit.detach()
                action = torch.bernoulli(logit)

                s, rew, done, info = self.env.step(action.squeeze().numpy().astype(np.int8))
                r_sum += rew
                s = transform(s).unsqueeze(dim=0)  # .to(device)
                rew = torch.Tensor([rew])  # .to(device)
                done = done or episode_length >= self.opt.max_episode_length

                #  add to trace - 1
                trace_a.append(action)
                trace_rew.append(rew)
                trace_aprob.append(logit)
                if done:
                    print("over, reward {}".format(r_sum))
                    r_sum = 0
                    episode_length = 0
                    # game over punishment
                    trace_rew[-1] = torch.Tensor([-200.])
                    break
            # add to trace - 2
            trace_s.append(s)
            # stack n-step
            # s[n_step+1, 3, width, height]
            # a[n_step, a_space]
            # rew[n_step]
            # a_prob[n_step]
            trace_s = torch.cat(tuple(trace_s), dim=0)
            zeros = torch.zeros((self.opt.n_step + 1,) + trace_s.size()[1:]).to(device)  # expand
            zeros[:trace_s.size(0)] += trace_s
            trace_s = zeros

            trace_a = torch.cat(tuple(trace_a), dim=0)
            zeros = torch.zeros((self.opt.n_step,) + trace_a.size()[1:]).to(device)  # expand
            zeros[:trace_a.size(0)] += trace_a
            trace_a = zeros

            trace_rew = torch.cat(tuple(trace_rew), dim=0)
            zeros = torch.zeros(self.opt.n_step).to(device)  # expand
            zeros[:trace_rew.size(0)] += trace_rew
            trace_rew = zeros

            trace_aprob = torch.cat(tuple(trace_aprob), dim=0)
            zeros = torch.zeros((self.opt.n_step,) + trace_aprob.size()[1:]).to(device)  # expand
            zeros[:trace_aprob.size(0)] += trace_aprob
            trace_aprob = zeros

            # submit trace to queue
            self.trace_queue.push((trace_s, trace_a, trace_rew, trace_aprob))

            if done:
                s = self.env.reset()
                s = transform(s).unsqueeze(dim=0)  # .to(device)
