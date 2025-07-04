# MoGe-2 Normal Estimation

<img src="..\assets\normal_comaprison.jpg">
<div align="center">
  <p style="text-align:center;">Qualitative comparison of normal estimation with <a href="https://github.com/prs-eth/marigold">Marigold</a> and <a href="https://github.com/YvanYin/Metric3D">Metric3D V2</a></p>
</div>

> NOTE: Normal estimation was implemented after the submission of the MoGe-2 paper and is therefore not included in the original publication. This feature required minimal additional effort, and we do not claim any novel technical contribution.

We added a lightweight convolutional head and trained the normal output using a squared angular loss:

$$
\mathcal L_{\rm normal} = {1\over |\mathcal M|}\sum_{i\in\mathcal M} \angle (\hat{\mathbf n}_i,\mathbf n_i)^2
$$

where $\hat{\mathbf{n}}_i$ is the predicted normal, $\mathbf{n}_i$ is the ground-truth normal, and $\mathcal{M}$ denotes the set of valid pixels. For convenience, we did not collect ground-truth normal maps for training. Instead, we derived surface normals from the depth map and camera intrinsics. The resulting estimates are visually and numerically satisfactory.
