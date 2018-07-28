import retro
import torch
from multiprocessing.managers import BaseManager
# from multiprocessing import Process
import torch.multiprocessing as mp
import os
from option import get_opt
from model import Actor, Learner, TraceQueue
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))


# env = retro.make(game='SonicAndKnuckles3-Genesis')
# sample = env.action_space.sample()
class RemoteManager(BaseManager): pass


def get_manager():
    RemoteManager.register('TraceQueue', TraceQueue)
    m = RemoteManager()
    m.start()
    return m


if __name__ == '__main__':
    opt = get_opt()
    # mp.set_start_method('spawn')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    # 进程共享数据
    m = get_manager()
    queue = m.TraceQueue()
    learner = Learner(opt, queue)
    actors = [Actor(opt, queue, learner), Actor(opt, queue, learner), Actor(opt, queue, learner)]
    processes = []
    for rank, a in enumerate(actors):
        p = mp.Process(target=a.performing, args=(rank,))
        p.start()
        processes.append(p)
    learner.learning()
    for p in processes:
        p.join()
