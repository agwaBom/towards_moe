# A Pytorch Implementation of "Towards Understanding Mixture of Experts in Deep Learning"

This is my implementation of "Towards Understanding Mixture of Experts in Deep Learning" which is accepted at NeurIPS 2022.

Which still has a lot of work to do.

**NOTE: I am not an author of the paper!!**

Below is the tsne visualization of synthetic data.
<p align = "center">
<img src="images/synthetic_cluster.png" width="500">
</p>
<p align = "center">
Figure 1. Each color denotes a cluster in synthetic data
</p>

<p align = "center">
<img src="images/synthetic_label.png" width="500">
</p>
<p align = "center">
Figure 2. Labels on each data point in synthetic data
</p>



## Performance
- Performance after 500 epoch

<p align = "center">
<img src="images/towards_moe_test_accuracy.png" width="1000">
<img src="images/towards_moe_test_loss.png" width="1000">
</p>
<p align = "center">
Figure 3. Accuracy and loss graph on each setting of model
</p>

|                    | Test accuracy (%) | Number of Filters |
|--------------------|-------------------|-------------------|
| Single (linear)    |              76.3 |               512 |
| Single (nonlinear) |              80.6 |               512 |
| MoE (linear)       |              96.2 |        128 (16*8) |
| MoE (nonlinear)    |              1.00 |        128 (16*8) |

<p align = "center">
<img src="images/linear_moe.gif" width="500">
</p>
<p align = "center">
Figure 4. A linear moe model that learns to dispatch data points to 8 experts.
</p>

<p align = "center">
<img src="images/non_linear_moe.gif" width="500">
</p>
<p align = "center">
Figure 5. A non-linear moe model that learns to dispatch data points to 8 experts
</p>

## Future work
### Model side
1. Add dispatch entropy evaluation
2. Support on Language & Image dataset
3. Replicate results on linear/non-linear MoE

### Dataset side
1. Fix synthetic data generation
    - Add cluster label for entropy evaluation
- On current version seems different from Figure 1 in the original paper
![](images/synthetic_data.png)


## Reference
~~~
@misc{https://doi.org/10.48550/arxiv.2208.02813,
  doi = {10.48550/ARXIV.2208.02813},
  url = {https://arxiv.org/abs/2208.02813},
  author = {Chen, Zixiang and Deng, Yihe and Wu, Yue and Gu, Quanquan and Li, Yuanzhi},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Towards Understanding Mixture of Experts in Deep Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
~~~