
import numpy as np
import random
class TorchSegTransform:
    """
    IMPORTANT: This version returns NUMPY arrays (HWC for image, HW for mask),
    because your dataset converts to torch and permutes to CHW after transforms.
    """
   # p_noise=0.1, noise_std=0.01
   # p_noise=0.2, noise_std=0.05
    def __init__(self, train=True, p_rot=0.5, p_h=0.5, p_v=0.5, p_noise=0.2, noise_std=0.05, debug=False):
        self.train = train
        self.p_rot = p_rot
        self.p_h = p_h
        self.p_v = p_v
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.debug = debug

    def __call__(self, image=None, mask=None, **kwargs):
        x = image
        y = mask

        # ensure numpy
        x = np.asarray(x)
        if y is not None:
            y = np.asarray(y)

        # normalize x to HWC
        sz=7
        if x.ndim == 3 and x.shape[0] == sz and x.shape[-1] != sz:   # CHW -> HWC
            x = np.transpose(x, (1, 2, 0))

        if not (x.ndim == 3 and x.shape[-1] == sz):
            raise ValueError(f"Unexpected image shape (need HWC with 7 bands): {x.shape}")

        # normalize mask to HW
        if y is not None:
            if y.ndim == 3 and y.shape[-1] == 1:   # H W 1
                y = y[..., 0]
            if y.ndim == 3 and y.shape[0] == 1:    # 1 H W
                y = y[0]
            if y.ndim != 2:
                raise ValueError(f"Unexpected mask shape (need HW): {y.shape}")

        if self.debug:
            print("[DEBUG] before augs: x", x.shape, "y", None if y is None else y.shape)

        # augs
        if self.train:
            if random.random() < self.p_rot:
                k = random.randint(0, 3)
                x = np.rot90(x, k, axes=(0, 1)).copy()
                if y is not None:
                    y = np.rot90(y, k, axes=(0, 1)).copy()

            if random.random() < self.p_h:
                x = np.flip(x, axis=1).copy()
                if y is not None:
                    y = np.flip(y, axis=1).copy()

            if random.random() < self.p_v:
                x = np.flip(x, axis=0).copy()
                if y is not None:
                    y = np.flip(y, axis=0).copy()

            if random.random() < self.p_noise:
                x = (x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std).astype(np.float32)

        # return numpy (dataset will convert to torch CHW)
        return {"image": x.astype(np.float32), "mask": y}
