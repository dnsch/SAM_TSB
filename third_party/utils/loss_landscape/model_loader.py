import os, sys
import cifar10.model_loader
import torch

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1] / "code"))
from src.models.time_series import samformer
from src.models.time_series.formers import patchtst


def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == "cifar10":
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net


def wideresnet_load(
    model_name,
    model_file,
    depth,
    width_factor,
    dropout,
    in_channels,
    labels,
    data_parallel=False,
):
    net = cifar10.model_loader.wideresnet_load(
        model_name,
        model_file,
        depth,
        width_factor,
        dropout,
        in_channels,
        labels,
        data_parallel,
    )
    return net


def samformer_load(
    node_num,
    input_dim,
    output_dim,
    num_channels,
    seq_len,
    hid_dim,
    horizon,
    use_revin,
    model_name,
    model_file=None,
    data_parallel=False,
):
    models = {
        "samformer": samformer.SAMFormer,
    }

    net = models[model_name](
        seq_len=seq_len,
        hid_dim=hid_dim,
        pred_len=horizon,
        use_revin=use_revin,
    )
    if data_parallel:
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if "state_dict" in stored.keys():
            net.load_state_dict(stored["state_dict"])
        else:
            net.load_state_dict(stored)

    if data_parallel:
        net = net.module

    net.eval()
    return net


def patchtst_load(
    enc_in,
    seq_len,
    pred_len,
    e_layers,
    n_heads,
    d_model,
    d_ff,
    dropout,
    fc_dropout,
    head_dropout,
    patch_len,
    stride,
    padding_patch,
    decomposition,
    kernel_size,
    individual,
    model_file=None,
    data_parallel=False,
    # Additional optional parameters with defaults
    max_seq_len=1024,
    d_k=None,
    d_v=None,
    norm="BatchNorm",
    attn_dropout=0.0,
    act="gelu",
    key_padding_mask="auto",
    padding_var=None,
    attn_mask=None,
    res_attention=True,
    pre_norm=False,
    store_attn=False,
    pe="zeros",
    learn_pe=True,
    pretrain_head=False,
    head_type="flatten",
    verbose=False,
):
    """
    Load PatchTST model with specified parameters.
    """
    net = patchtst.PatchTST(
        enc_in=enc_in,
        seq_len=seq_len,
        pred_len=pred_len,
        e_layers=e_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        fc_dropout=fc_dropout,
        head_dropout=head_dropout,
        patch_len=patch_len,
        stride=stride,
        padding_patch=padding_patch,
        decomposition=decomposition,
        kernel_size=kernel_size,
        individual=individual,
        max_seq_len=max_seq_len,
        d_k=d_k,
        d_v=d_v,
        norm=norm,
        attn_dropout=attn_dropout,
        act=act,
        key_padding_mask=key_padding_mask,
        padding_var=padding_var,
        attn_mask=attn_mask,
        res_attention=res_attention,
        pre_norm=pre_norm,
        store_attn=store_attn,
        pe=pe,
        learn_pe=learn_pe,
        pretrain_head=pretrain_head,
        head_type=head_type,
        verbose=verbose,
    )

    if data_parallel:
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if "state_dict" in stored.keys():
            net.load_state_dict(stored["state_dict"])
        else:
            net.load_state_dict(stored)

    if data_parallel:
        net = net.module

    net.eval()
    return net
