import torch
import torch.distributions.uniform as uniform
import argparse

torch.manual_seed(12)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-data_size", default=1000, help="output length: data size * cluster number")
    parser.add_argument("-P", default=4, help="(P) number of patches")
    parser.add_argument("-K", default=4, help="(K) cluster number")
    parser.add_argument("-d", default=50, help="(d) dimension of each patch")
    parser.add_argument("-alpha", default=(0.5, 2))
    parser.add_argument("-beta", default=(1, 2))
    parser.add_argument("-gamma", default=(1, 2))
    parser.add_argument("-abscon", default=1, help="absolute constant")

    # opt.parsed_name
    opt = parser.parse_args()

    K_all_data = list()

    K_data = list()

    # Initialize k
    k = torch.rand(opt.d)
    k = k/k.norm(p=2)

    # Uniformly draw a pair (k, k') with k != k' from {1,...,K}
    for i in range(0, opt.K*2):
        # Each cluster k has a label signal vector vk and a cluster-center signal vector ck
        # vk and ck are orthogonal, 2_norm(vk) and 2_norm(ck) == 1
        # vk, ck are 50 dimension vector
        #k1 = torch.nn.init.orthogonal_(torch.rand(2, 50))
        #k2 = torch.nn.init.orthogonal_(torch.rand(2, 50))
        #
        #assert not torch.equal(k1, k2), "k1, k2 are equal"

        # (deprecated: init.orthogonal_() already creates l2 norm of 1) l2(vk) == l2(ck) == 1
        #k1[0]=k1[0]/k1[0].norm(p=2)
        #k1[1]=k1[1]/k1[1].norm(p=2)
        #k2[0]=k2[0]/k2[0].norm(p=2)
        #k2[1]=k2[1]/k2[1].norm(p=2)

        K_data.append(k) # Past_k
        k = torch.rand(opt.d)
        k = k/k.norm(p=2)
        for k_past in K_data:
            k = k - (k@k_past*k_past)
        k = k/k.norm(p=2)

    K_data = [(K_data[i], K_data[i+(len(K_data)//2)]) for i in range(0, len(K_data)//2)]

    for k in K_data:
        # Generate the label y and epsilon uniformly (Rademacher distribution)
        y = [1 if i > 0.5 else -1 for i in torch.rand(opt.data_size)]
        eps = [1 if i > 0.5 else -1 for i in torch.rand(opt.data_size)]

        # Independently generate random variables α, β, γ from distribution Dα, Dβ, Dγ
        # C1 = O(M**2) and C2 = O~(d**−0.005)
        # Given two sequences {x_n} and {y_n}, we denote x_n = O(y_n) if |x_n| ≤ C1|y_n| for some absolute positive constant C1
        # basic_usage = uniform.Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        D_alpha = uniform.Uniform(torch.Tensor([opt.alpha[0]]), torch.Tensor([opt.alpha[1]]))
        D_beta = uniform.Uniform(torch.Tensor([opt.beta[0]]), torch.Tensor([opt.beta[1]]))
        D_gamma = uniform.Uniform(torch.Tensor([opt.gamma[0]]), torch.Tensor([opt.gamma[1]]))


        alpha = [D_alpha.sample() for _ in range(0, opt.data_size)]
        beta = [D_beta.sample() for _ in range(0, opt.data_size)]
        gamma = [D_gamma.sample() for _ in range(0, opt.data_size)]

        # Now, we all have ingredients haha :D
        # y, eps, alpha, beta, gamma, k_past(K_data[-1]), k_current(k)

        # Generate various signals
        feature_signals = [y[i] * alpha[i] * k[0] for i in range(0, opt.data_size)]
        cluster_center_signals = [beta[i] * k[1] for i in range(0, opt.data_size)]
        feature_noises = [eps[i] * gamma[i] * k[0] for i in range(0, opt.data_size)]

        # The rest of P − 3 patches are Gaussian noises that are independently 
        # drawn from N(0,(σ_p**2/d) · Id) where σ_p is an absolute constant.

        # Id (random noise) is not detailed in paper
        random_noises = torch.normal(mean=0, std=((opt.abscon**2)/opt.d), size=(opt.data_size, opt.P-3, opt.d))

        # [data_size, patches, dimension]
        patches = [torch.cat((torch.stack([feature_signals[i], cluster_center_signals[i], feature_noises[i]], dim=0), random_noises[i]), dim=0) for i in range(0, opt.data_size)]
        K_all_data.append(torch.stack(patches, dim=0))

    # If we want to save per cluster
    for i, d in enumerate(K_all_data):
        torch.save(d, './cluster_%d_data.pt'%i)

    # If we want to save all in one
    torch.save(torch.cat(K_all_data, dim=0), './cluster_all_data.pt')

    return



if __name__ == "__main__":
    main()