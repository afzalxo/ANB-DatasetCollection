import random


class Searchables:
    def __init__(self):
        self.num_blocks = 5  # 7
        # self.strides = [1, 2, 2, 2, 1, 2, 1]
        self.strides = [1, 1, 1, 1, 1, 1, 1]
        self.ich = [32, 16, 24, 40, 80, 112, 192]
        self.och = [16, 24, 40, 80, 112, 192, 320]
        # self.ich = [16, 24, 32, 32, 80, 112, 192]
        # self.och = [24, 32, 48, 80, 112, 192, 320]
        self.bchoices = ["MB" for i in range(self.num_blocks)]
        self.kernel_sizes = [3, 5]
        self.exp = [1, 4, 6]
        self.layers = [1, 2, 3]  # Choices are 1,2,3 for all blocks except 6th
        # self.squeeze_excite_options = [True, False]
        self.squeeze_excite_options = [False]

    def searchable_from_conf(self, _conf):
        layer_confs = []
        bchoices = random.choices(self.bchoices, k=self.num_blocks)
        for i in range(self.num_blocks):
            conf = [
                bchoices[i],
                _conf[0][i],
                _conf[1][i],
                self.strides[i],
                self.ich[i],
                self.och[i],
                _conf[2][i],
                _conf[3][i],
            ]
            layer_confs.append(conf)
        return layer_confs

    def random_searchable(self):
        kchoices = random.choices(self.kernel_sizes, k=self.num_blocks)
        echoices = random.choices(self.exp, k=self.num_blocks)
        lchoices = random.choices(self.layers, k=self.num_blocks)
        # lchoices[-2] = random.choice([1, 2, 3, 4])  # 6th block to conform to effnetb0
        sechoices = random.choices(self.squeeze_excite_options, k=self.num_blocks)
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def efficientnet_b0_conf(self):
        # Used efficientnet_b0 as baseline architecture to compared searched results against
        kchoices = [3, 3, 5, 3, 5, 5, 3]
        echoices = [1, 6, 6, 6, 6, 6, 6]
        lchoices = [1, 2, 2, 3, 3, 4, 1]
        sechoices = [True for i in range(self.num_blocks)]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def custom_searchable(self, e, k, la, se):
        return self.searchable_from_conf([e, k, la, se])

    def n_random_searchables(self, n, seed):
        random.seed(seed)
        searchables = [self.random_searchable() for _ in range(n)]
        return searchables

    def test_searchable(self):
        kchoices = [5, 5, 5, 5, 5, 5, 5]
        echoices = [6, 6, 6, 6, 6, 6, 6]
        lchoices = [3, 3, 3, 3, 3, 4, 3]
        sechoices = [True for i in range(self.num_blocks)]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_zcu102_a(self):
        # Searched result zcu102 model a with acc = 77.698
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/6sf5epkw?workspace=user-europa1610
        echoices = [4, 6, 6, 6, 6, 6, 6]
        kchoices = [3, 5, 5, 5, 5, 5, 5]
        lchoices = [2, 1, 3, 3, 3, 4, 3]
        sechoices = [False, False, False, False, False, False, True]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_zcu102_t(self):
        # Searched result zcu102 model b with acc = 76.602
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/22r1luie?workspace=user-europa1610
        echoices = [1, 6, 6, 6, 6, 6, 6]
        kchoices = [3, 3, 5, 3, 3, 5, 5]
        lchoices = [1, 2, 3, 1, 3, 4, 3]
        sechoices = [False, False, False, False, True, False, False]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_zcu102_tt(self):
        # Searched result zcu102 model c with acc = 75.048
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/d03umktt?workspace=user-europa1610
        echoices = [1, 6, 6, 6, 6, 6, 6]
        kchoices = [3, 3, 5, 5, 5, 5, 5]
        lchoices = [2, 1, 3, 3, 2, 4, 1]
        sechoices = [False, False, False, False, False, False, False]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_vck190_a(self):
        # Searched result vck190 model a with acc = 77.56800079
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/ys2vsx8i?workspace=user-europa1610
        echoices = [1, 6, 6, 6, 6, 6, 6]
        kchoices = [5, 3, 5, 5, 5, 5, 5]
        lchoices = [1, 2, 3, 3, 2, 4, 3]
        sechoices = [False, False, False, True, False, True, True]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_vck190_t(self):
        # Searched result vck190 model b with acc = 76.6559906
        # wandb eval run https://wandb.ai/europa1610/NASBenchFPGA/runs/vaxavbx1?workspace=user-europa1610
        echoices = [1, 6, 6, 6, 6, 6, 6]
        kchoices = [3, 3, 5, 5, 3, 5, 5]
        lchoices = [1, 2, 2, 3, 2, 4, 2]
        sechoices = [False, False, False, False, False, True, False]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_a100_a(self):
        # Searched result A100-40GiB model a with acc = 77.81
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/kfqttvsx?workspace=user-europa1610
        echoices = [1, 6, 6, 6, 6, 6, 6]
        kchoices = [3, 5, 5, 5, 5, 5, 5]
        lchoices = [2, 2, 3, 1, 3, 4, 3]
        sechoices = [False, False, False, True, True, True, True]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_a100_t(self):
        # Searched result A100-40GiB model b with acc = 76.50999451
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/x12gms30?workspace=user-europa1610
        echoices = [1, 4, 6, 6, 6, 6, 6]
        kchoices = [3, 3, 5, 5, 5, 5, 5]
        lchoices = [1, 1, 2, 3, 2, 4, 2]
        sechoices = [False, False, False, True, False, True, True]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_3090_a(self):
        # Searched result rtx3090 model a with acc = 77.35799
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/07hez0zj?workspace=user-europa1610
        echoices = [1, 4, 6, 6, 6, 6, 6]
        kchoices = [5, 5, 5, 5, 5, 5, 5]
        lchoices = [1, 2, 3, 3, 3, 4, 2]
        sechoices = [True, False, False, True, True, True, True]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_3090_t(self):
        # Searched result same as that of A100 model b
        return self.effnet_a100_t()

    def effnet_tpuv3_a(self):
        # Searched result TPUv3 model a with acc = 77.921
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/w178pu9g?workspace=user-europa1610
        echoices = [1, 4, 6, 6, 6, 6, 6]
        kchoices = [5, 5, 5, 5, 5, 5, 5]
        lchoices = [3, 3, 3, 3, 3, 4, 3]
        sechoices = [False, True, True, False, True, True, False]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])

    def effnet_tpuv3_t(self):
        # Searched result TPUv3 model b with acc = 77.348
        # wandb eval run at https://wandb.ai/europa1610/NASBenchFPGA/runs/f8q699br?workspace=user-europa1610
        echoices = [1, 6, 6, 6, 6, 6, 6]
        kchoices = [5, 5, 5, 5, 5, 5, 5]
        lchoices = [1, 2, 3, 3, 3, 3, 3]
        sechoices = [False, True, False, False, True, True, False]
        return self.searchable_from_conf([echoices, kchoices, lchoices, sechoices])


class FBNetSearchables:
    def __init__(self):
        self.num_blocks = 7
        self.layer_choices = [1, 2, 3, 4]
        self.num_layers = [1, 4, 4, 4, 4, 4, 1]
        """
        self.strides = [1, 2, 2, 2, 1, 2, 1]
        self.ich = [32, 16, 24, 40, 80, 112, 192]
        self.och = [16, 24, 40, 80, 112, 192, 320]
        self.bchoices = ["MB" for i in range(self.num_blocks)]
        self.kernel_sizes = [3, 5]
        self.exp = [1, 4, 6]
        self.layers = [1, 2, 3]  # Choices are 1,2,3 for all blocks except 6th
        self.squeeze_excite_options = [True, False]
        """
        self.primitives = [
            "k3_e1",
            "k3_e1_g2",
            "k3_e3",
            "k3_e6",
            "k5_e1",
            "k5_e1_g2",
            "k5_e3",
            "k5_e6",
            "skip",
        ]

    def random_searchable_baseline(self):
        lchoices = [1, 4, 4, 4, 4, 4, 1]
        return random.choices(list(range(len(self.primitives))), k=sum(lchoices)), lchoices

    def random_searchable_highvar(self):
        lchoices = random.choices(self.layer_choices, k=self.num_blocks)
        lchoices[0] = random.choice([1, 2, 3])
        return random.choices(list(range(len(self.primitives))), k=sum(lchoices)), lchoices
