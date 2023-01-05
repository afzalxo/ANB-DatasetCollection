import sys
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json
# import json
from Worker import Worker_Simulated
import numpy as np
import logging
from multiprocessing import Process, Queue
import random
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

sys.path.append('/home/aahmadaa/NASBenchFPGA/surrogates')
import AccelNB as ANB

def consume(worker, surrogate_model, actions_index, results_queue):
    worker.ObtainAcc(surrogate_model, actions_index)
    results_queue.put(worker)


class RandomSearch(object):
    def __init__(self, ss_configspace_path, arch_epochs, episodes, surrogate_model):
        with open(ss_configspace_path, 'r') as fh:
            json_string = fh.read()
        self.surrogate_model = surrogate_model
        self.ss_configspace = json.read(json_string)
        self.arch_epochs = arch_epochs
        self.episodes = episodes

    def random_sample(self):
        return self.ss_configspace.sample_configuration()

    def multi_solve_environment(self):
        workers_top20 = []

        for arch_epoch in range(self.arch_epochs):
            results_queue = Queue()
            processes = []

            for episode in range(self.episodes):
                actions_index = self.random_sample()
                worker = Worker_Simulated()#self.surrogate_model)
                process = Process(target=consume, args=(worker, self.surrogate_model, actions_index, results_queue))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            workers = []
            for episode in range(self.episodes):
                worker = results_queue.get()
                workers.append(worker)

            # sort worker retain top20
            workers_total = workers_top20 + workers
            workers_total.sort(key=lambda worker: worker.acc, reverse=True)
            workers_top20 = workers_total[:20]
            top1_acc = workers_top20[0].acc
            top5_avg_acc = np.mean([worker.acc for worker in workers_top20[:5]])
            top20_avg_acc = np.mean([worker.acc for worker in workers_top20])
            logging.info(
                'arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f}'.format(
                    arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc))
            # for i in range(5):
            #    print(workers_top20[i].genotype)


def main():
    configspace_path = './nas_algos/configspace.json'
    surrogate_model_dir = '/home/aahmadaa/NASBenchFPGA/surrogates/anb_models/xgb/20230101-174844-4'
    surr_model = ANB.load_model(surrogate_model_dir)
    sd = RandomSearch(configspace_path, 10, 10, surr_model)
    sd.multi_solve_environment()


if __name__ == '__main__':
    main()
