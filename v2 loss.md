# v2 loss
## loss修改需求
1. 在 ddpm_with_edge.py 中的 p_losses() function，有 edge_map as input argument;
2. 用 model_output_ 和 edge_map 计算 edge_loss。用 MSE（也叫L2）计算
3. 将 edge_loss * 0.1（这个要多次training尝试，从0.1开始试），加到总loss中