import time
import shutil
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist

import mmcv
from mmcls.models.losses import accuracy
from mmcv.runner import get_dist_info


@torch.no_grad()
# 根据策略选择一个logits供后续使用
def select_logits_to_idx(x1: list, x2: list, mode: str):
    logits1 = torch.from_numpy(np.vstack(x1))
    logits2 = torch.from_numpy(np.vstack(x2))
    num = logits1.shape[0]
    if mode == 'first':
        idx = torch.zeros(num, 1)
    elif mode == 'second':
        idx = torch.ones(num, 1)
    elif mode == 'entropy':
        ent1 = - (logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
        ent2 = - (logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
        idx = ent1>=ent2
    elif mode == 'confidence':
        con1 = logits1.softmax(1).max(1, keepdim=True)[0]
        con2 = logits2.softmax(1).max(1, keepdim=True)[0]
        idx = con1<=con2
    elif mode == 'var':
        v1 = logits1.var(dim=1, keepdim=True)
        v2 = logits2.var(dim=1, keepdim=True)
        idx = v1<=v2
    else:
        raise NotImplementedError(f"No such select mode {mode}")
    
    # generate the selected logits
    logits = logits1 * idx.logical_not() + logits2 * idx
    logits = logits.numpy()
    return [logits[i] for i in range(logits.shape[0])], idx


@torch.no_grad()
def tackle_img_from_idx(img1: dict, img2: dict, idx: torch.tensor) -> dict:
    ith = idx.clone()
    while img1['img'].dim() > ith.dim():
        ith = ith.unsqueeze(-1)
    img1['img'] = img1['img'] * ith.logical_not() + img2['img'] * ith
    for i in range(idx.shape[0]):
        if idx[i]==1:
            img1['img_metas'].data[0][i] = img2['img_metas'].data[0][i]
    return img1


@torch.no_grad()
# 对两个输出的熵、置信度和方差进行加权得到权重，然后使用得到的权重对图像进行加权融合
def fuse_img_from_logits(x1: list, x2: list, img1: dict, img2: dict, mode: str) -> dict:
    logits1 = torch.from_numpy(np.vstack(x1))
    logits2 = torch.from_numpy(np.vstack(x2))
    if mode == 'entropy_fuse':
        ent1 = - (logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
        ent2 = - (logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
        w = torch.cat((ent2, ent1), 1).softmax(1)
    elif mode == 'confidence_fuse':
        con1 = logits1.softmax(1).max(1, keepdim=True)[0]
        con2 = logits2.softmax(1).max(1, keepdim=True)[0]
        w = torch.cat((con1, con2), 1).softmax(1)
    elif mode == 'var_fuse':
        v1 = logits1.var(dim=1, keepdim=True)
        v2 = logits2.var(dim=1, keepdim=True)
        w = torch.cat((v1, v2), 1).softmax(1)
    else:
        raise NotImplementedError(f"No such select mode {mode}")
    
    # generate the selected logits
    while img1['img'].dim() > w.dim():
        w = w.unsqueeze(-1)
    img1['img'] = img1['img'] * w[:, :1] + img2['img'] * w[:, 1:]
    return img1


@torch.no_grad()
# 对x1和x2中的logits进行加权融合
def ensemble_from_logits(x1: list, x2: list, mode: str):
    logits1 = torch.from_numpy(np.vstack(x1))
    logits2 = torch.from_numpy(np.vstack(x2))
    if mode == 'sum':
        logits = logits1 + logits2
    # 根据熵值加权相加计算
    elif mode == 'entropy_sum':
        ent1 = - (logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
        ent2 = - (logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
        logits = logits1 * ent2  + logits2 * ent1
    # 根据置信度进行加权相加
    elif mode == 'confidence_sum':
        con1 = logits1.softmax(1).max(1, keepdim=True)[0]
        con2 = logits2.softmax(1).max(1, keepdim=True)[0]
        logits = logits1 * con1  + logits2 * con2
    else:
        raise NotImplementedError(f"No such select mode {mode}")
    logits = logits.numpy()
    return [logits[i] for i in range(logits.shape[0])]


def single_gpu_test_ensemble(
        model,
        data_loader,
        mode,
        show=False,
        out_dir=None,
        **show_kwargs
    ):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, (data1, data2)  in enumerate(data_loader):
        with torch.no_grad():
            result1 = model(return_loss=False, **data1)
            result2 = model(return_loss=False, **data2)
            if 'sum' in mode:
                result = ensemble_from_logits(result1, result2, mode)
                data = data1
            elif 'fuse' in mode:
                data = fuse_img_from_logits(result1, result2, data1, data2, mode)
                result = model(return_loss=False, **data)
            else:
                result, idx = select_logits_to_idx(result1, result2, mode)
                data = tackle_img_from_idx(data1, data2, idx)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test_ensemble(model, data_loader, mode, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

        This method tests model with multiple gpus and collects the results
        under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
        it encodes results to gpu tensors and use gpu communication for results
        collection. On cpu mode it saves the results on different gpus to 'tmpdir'
        and collects them by the rank 0 worker.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader for 2 dataset.
            mode (str): criterion for unsemble
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode.
            gpu_collect (bool): Option to use either gpu or cpu to collect results.

        Returns:
            list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        # if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
        #     raise OSError((f'The tmpdir {tmpdir} already exists.',
        #                    ' Since tmpdir will be deleted after testing,',
        #                    ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier() 
    for i, (data1, data2) in enumerate(data_loader):
        with torch.no_grad():
            result1 = model(return_loss=False, **data1)
            result2 = model(return_loss=False, **data2)
            if 'sum' in mode:
                result = ensemble_from_logits(result1, result2, mode)
                data = data1
            elif 'fuse' in mode:
                data = fuse_img_from_logits(result1, result2, data1, data2, mode)
                result = model(return_loss=False, **data)
            else:
                result, idx = select_logits_to_idx(result1, result2, mode)
                data = tackle_img_from_idx(data1, data2, idx)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
