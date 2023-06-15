import torch

import triton
import triton.language as tl
from einops import repeat, rearrange


def exists(x):
    return x is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fps_kernel(x_ptr, dist_ptr, out_ptr,
               N, M, C,
               stride_xb, stride_xn, stride_xc,
               stride_dist_b, stride_dist_n,
               stride_out_b, stride_out_m,
               BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    # x_ptr: (b, n, 3)
    # dist_ptr: (b, n)
    # out_ptr: (b, m) store index of sampled points
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_b = pid // num_pid_n

    start_ptr_xb = x_ptr + pid_b * stride_xb
    start_ptr_dist_b = dist_ptr + pid_b * stride_dist_b
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)

    idx = tl.load(out_ptr + pid_b * stride_out_b)  # current idx
    for i in range(1, M):
        mask_c = offs_c < C
        xi = tl.load(start_ptr_xb + idx * stride_xn + offs_c[None, :] * stride_xc,
                     mask=mask_c[None, :], other=0.)  # (1, 4)

        x_ptrs = start_ptr_xb + (offs_n[:, None] * stride_xn + tl.arange(0, BLOCK_C)[None, :] * stride_xc)  # (n, 4)
        dist_ptrs = start_ptr_dist_b + offs_n * stride_dist_n
        # initialize memory to store max dist and idx in each block
        max_dist_n = tl.zeros((BLOCK_N,), dtype=tl.float32)
        idx_n = tl.zeros((BLOCK_N,), dtype=tl.int32)
        for n in range(0, tl.cdiv(N, BLOCK_N)):
            # inner loop: given idx, compute dists
            mask = offs_n < N - n * BLOCK_N
            x = tl.load(x_ptrs, mask=mask[:, None] & mask_c[None, :], other=0.)  # (n, 4)
            # compute distance
            dist = tl.sqrt(tl.sum((x - xi) * (x - xi), axis=1))

            # tl.atomic_min(dist_ptrs, dist, mask=mask)  # save to DRAM
            dists = tl.load(dist_ptrs, mask=mask, other=-float('inf'))
            dists = tl.minimum(dist, dists)
            tl.store(dist_ptrs, dists, mask=mask)  # save to DRAM

            # find max dist and index: get the lowest index if multiple maximum values
            _max_dist = tl.max(dists, axis=0)
            _idx = tl.where(dists == _max_dist, tl.arange(0, BLOCK_N), BLOCK_N)
            _idx = tl.min(_idx, axis=0) + n * BLOCK_N
            # _idx = tl.argmax(dists, axis=0) + n * BLOCK_N

            # reserve max dist and index
            masked_dist_n = tl.where(tl.arange(0, BLOCK_N) == n, _max_dist, 0.)
            masked_idx_n = tl.where(tl.arange(0, BLOCK_N) == n, _idx, 0)

            # update max dist
            max_dist_n += masked_dist_n
            idx_n += masked_idx_n

            # advance pointers
            x_ptrs += BLOCK_N * stride_xn
            dist_ptrs += BLOCK_N * stride_dist_n

        # update idx
        # i_max_dist_n = tl.argmax(max_dist_n, axis=0)
        idx_n = tl.where(max_dist_n == tl.max(max_dist_n), idx_n, N)
        idx = tl.min(idx_n, axis=0)

        # update idx
        tl.store(out_ptr + pid_b * stride_out_b + i * stride_out_m, idx)


def fps_triton(x, n_sample, start_idx: int = None):
    # x: (b, n, 3)
    # n_sample: number of points to sample
    # return: (b, n_sample) store index of sampled points
    b, n, c = x.shape
    if exists(start_idx):
        out = torch.full((b, n_sample), start_idx, dtype=torch.int32, device=x.device)
    else:
        out = torch.randint(0, n, (b, n_sample), dtype=torch.int32, device=x.device)

    dists = torch.full((b, n), float('inf'), dtype=torch.float32, device=x.device)
    stride_xb, stride_xn, stride_xc = x.stride()
    stride_out_b, stride_out_m = out.stride()
    stride_dist_b, stride_dist_n = dists.stride()

    grid = lambda meta: (b * triton.cdiv(n, meta['BLOCK_N']),)
    fps_kernel[grid](x, dists, out,
                     n, n_sample, c,
                     stride_xb, stride_xn, stride_xc,
                     stride_dist_b, stride_dist_n,
                     stride_out_b, stride_out_m,
                     BLOCK_C=triton.next_power_of_2(c))

    return out.long()


def farthest_point_sampling(x: torch.Tensor, n_sample: int, start_idx: int = None):
    # x: (b, n, 3)
    b, n = x.size()[:2]
    assert n_sample <= n, "not enough points to sample"

    if n_sample == n:
        return repeat(torch.arange(n_sample, dtype=torch.long, device=x.device), 'm -> b m', b=b)

    # start index
    if exists(start_idx):
        sel_idx = torch.full((b, n_sample), start_idx, dtype=torch.long, device=x.device)
    else:
        sel_idx = torch.randint(n, (b, n_sample), dtype=torch.long, device=x.device)

    cur_x = rearrange(x[torch.arange(b), sel_idx[:, 0]], 'b c -> b 1 c')
    min_dists = torch.full((b, n), dtype=x.dtype, device=x.device, fill_value=float('inf'))
    for i in range(1, n_sample):
        # update distance
        dists = torch.linalg.norm(x - cur_x, dim=-1)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        idx_farthest = torch.max(min_dists, dim=-1).indices
        sel_idx[:, i] = idx_farthest
        cur_x[:, 0, :] = x[torch.arange(b), idx_farthest]

    return sel_idx
