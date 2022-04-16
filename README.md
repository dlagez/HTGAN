引用了这篇文章的代码，简单的整理了一下。

目前准备先在公共的数据集上面运行，然后自己准备数据集训练。

fork from : `https://github.com/summitgao/HC_ADGAN`，感谢作者开源。



> This repo holds the codes of our paper:
>
> **Adaptive Dropblock Enhanced GenerativeAdversarial Networks for Hyperspectral Image Classification**
>
> in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 6, pp. 5040-5053, June 2021, doi: 10.1109/TGRS.2020.3015843.
>
> The demo has not been well organized. Please contact me if you meet any problems.
>
> Please cite our paper if you use our codes. Thanks!
>
> If you have any queries, please do not hesitate to contact me ( *gaofeng AT ouc.edu.cn* ).
>
> More codes can be obtained from *http://feng-gao.cn*
>
> 
>
> **Requirements:** Python >= 3.6, PyTorch and torchvision 
>
> The PaviaU.mat and PaviaU_gt.mat stands for the Pavia University dataset and it's corresponding labels respectively, and the PCU.mat is the result of the Pavia University dataset after PCA.
>



在我将作者的代码导入到本地运行时，报了一些错误，是一些由于环境不一样（pytorch版本不一样）导致的问题：

使用：`git diff Demo_gan.py`查看修改代码记录。

解决方法参考pytorch官方解决方法：[link](https://discuss.pytorch.org/t/runtimeerror-set-sizes-contiguous-is-not-allowed-on-tensor-created-from-data-or-detach-in-pytorch-1-1-0/44208) 有兴趣可以看一下。

```text
@@ -542,13 +540,14 @@ for epoch in range(1, opt.niter + 1):
             noise.resize_(batch_size, nz, 1, 1)
             noise.normal_(0, 1)
             noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))
-
-            noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))
+            with torch.no_grad():^M
+                noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))^M
 
             #label = np.random.randint(0, nb_label, batch_size)
-            label = np.full(batch_size, nb_label)
-
-            f_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
+            label = np.full(batch_size, nb_label)^M
+            with torch.no_grad():^M
+                # f_label.data.resize_(batch_size).copy_(torch.from_numpy(label))^M
+                f_label.resize_(batch_size).copy_(torch.from_numpy(label))^M

```



由于作者将所有代码都打包到一个文件，所以我将代码重新组织了一下。目前并没有做任何的改动。
