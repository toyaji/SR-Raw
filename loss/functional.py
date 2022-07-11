import torch
from torch._C import ErrorReport
import torch.nn.functional as F
import torchvision.transforms.functional as TF

__all__ = ['contextual_loss', 'contextual_bilateral_loss']


LOSS_TYPES = ['cosine', 'l1', 'l2']


##### losses #####

def contextual_loss(x: torch.Tensor,
                    y: torch.Tensor,
                    band_width: float = 0.5,
                    loss_type: str = 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features. ['l1', 'l2', 'cosine']
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    if loss_type == 'cosine':
        dist_raw = _compute_cosine_distance(x, y)
    elif loss_type == 'l1':
        dist_raw = _compute_l1_distance(x, y)
    elif loss_type == 'l2':
        dist_raw = _compute_l2_distance(x, y)
    else:
        raise AttributeError("Proper loss type should be given among ['l1, 'l2']")

    dist_tilde = _compute_relative_distance(dist_raw)
    cx = _compute_cx(dist_tilde, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

    return cx_loss


# TODO: Operation check
def contextual_bilateral_loss(x: torch.Tensor,
                              y: torch.Tensor,
                              weight_sp: float = 0.1,
                              band_width: float = 1.,
                              loss_type: str = 'cosine'):
    """
    Computes Contextual Bilateral (CoBi) Loss between x and y,
        proposed in https://arxiv.org/pdf/1905.05169.pdf.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features. ['l1', 'l2', 'cosine']
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper).
    k_arg_max_NC : torch.Tensor
        indices to maximize similarity over channels.
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    # spatial loss - it is a regulation or penalty to more far distant points
    # TODO 이거는 이미지 사이즈 한번 정해지면 상수라서 이 부분 연산 계속 안하게 고칠 필요 있겠네
    grid = _compute_meshgrid(x.shape).to(x.device)
    dist_raw = _compute_l2_distance(grid, grid)
    dist_tilde = _compute_relative_distance(dist_raw)
    cx_sp = _compute_cx(dist_tilde, band_width)

    # feature loss
    if loss_type == 'cosine':
        dist_raw = _compute_cosine_distance(x, y)
    elif loss_type == 'l1':
        dist_raw = _compute_l1_distance(x, y)
    elif loss_type == 'l2':
        dist_raw = _compute_l2_distance(x, y)

    dist_tilde = _compute_relative_distance(dist_raw)
    cx_feat = _compute_cx(dist_tilde, band_width)

    # combined loss
    cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


def unalign_loss(x: torch.Tensor, 
                 y: torch.Tensor, 
                 tol: int = 16,
                 stride: int = 1, 
                 loss_type='l1'):
    """
    Capture unalignment of the real raw data.
    The most of the code follows the ways of 'zoom-to-learn' paper:
        https://github.com/ceciliavision/zoom-learn-zoom

    Parameters
    ---
    x : torch.Tensor
        features o shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    tol : int
        tolerance(pixel buffer) for boundary unalingment between x and y.
    stride : int
        stride size for making aligned features by translating input.
    loss_type : str, optional
        a loss type to measure the degree of unlignment of given paris. ['l1', 'l2', 'cosine']

    Returns
    ---
    unalign_loss : torch.Tensor
        unalign loss between x and y 
    """
    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    num_tiles = int(tol*2/stride) * int(tol*2/stride)
    translation_range = [[-i,-j] for i in range(0,(tol*2),stride) 
                                 for j in range(0,(tol*2),stride)]
    
    translated_y = []
    for i_x, i_y in zip(x, y):
        argmin_y, _ = _get_argmin_translated_target_y(i_x, i_y, num_tiles, tol*2, translation_range)
        translated_y.append(argmin_y)
    y = torch.stack(translated_y)
    
    if x.size() != y.size():
        x = x[:,:,:-tol*2, :-tol*2]

    if loss_type == 'l1':
        loss = F.l1_loss(x, y)
    elif loss_type == 'l2':
        loss = F.mse_loss(x, y)
    elif loss_type == 'cosine':
        loss = F.cosine_embedding_loss(x, y, reduce=True)

    return loss




##### funcs for losses #####

def _compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def _compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def _compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist


# TODO: Considering avoiding OOM.
def _compute_l1_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.reshape(N, C, -1)
    y_vec = y.reshape(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
    dist = dist.clamp(min=0.)

    return dist


# TODO: Considering avoiding OOM.
def _compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.reshape(N, C, -1)
    y_vec = y.reshape(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True) 
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)
    
    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s
    dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
    dist = dist.clamp(min=0.)

    return dist


def _compute_meshgrid(shape):
    # Constant mesh grid acts as like spatial penalty mask.
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid


def _get_argmin_translated_target_y(x, y, num_tiles, tol, translation_range):
    # Make tiles to compare of all possible translation. 
    # TODO OOM 해결하려면 여기서 부터...
    C, H, W = x.size()
    xs = torch.tile(x, [num_tiles, 1, 1, 1])
    ys = torch.tile(y, [num_tiles, 1, 1, 1])
    ys = _get_translated_tiles(ys, (H, W), translation_range)
    ys = ys[:, :, :H-tol, :W-tol]
    xs = xs[:, :, :H-tol, :W-tol]
    
    # calculate dist between translated ones and origin
    assert xs.size() == ys.size(), "x and y have different sizes"
    dist = torch.mean(ys - xs, [1, 2, 3], keepdim=True)
    argmin_tile_indice = dist.min(0).indices.squeeze()
    argmin_y = ys[argmin_tile_indice]
    return argmin_y, argmin_tile_indice


def _get_translated_tiles(ys, size: tuple, translation_range):
    # translated along the tol range
    H, W = size
    for _, (i, j) in enumerate(translation_range):
        ys[_] = TF.affine(ys[_], 0, [i, j], 1, 0, 
            TF.InterpolationMode.BILINEAR)
    return ys
