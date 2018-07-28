import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue, Queue
from option import get_opt
from model import Actor, Learner, QManeger
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))


if __name__ == '__main__':
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    processes = []
    # data communication
    q_trace = Queue(maxsize=300)
    q_batch = Queue(maxsize=3)
    q_manager = QManeger(opt, q_trace, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()
    processes.append(p)

    learner = Learner(opt, q_batch)  # inner shared network was used by actors.
    actors = [Actor(opt, q_trace, learner), Actor(opt, q_trace, learner), Actor(opt, q_trace, learner)]
    for rank, a in enumerate(actors):
        p = mp.Process(target=a.performing, args=(rank,))
        p.start()
        processes.append(p)

    learner.learning()
    for p in processes:
        p.join()
